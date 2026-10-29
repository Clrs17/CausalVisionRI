# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Phase-2 training: DV24 regression pretrain -> RI classification finetune
(focal loss), LOYO (leave-one-year-out) evaluation by storm.

Causal losses (see models.CausalLossTerms):
  - adjacency CE: last-layer GATv2 attention over each node's incoming prior
    edges (all lags) matched to the prior |coef| distribution (B2)
  - sign consistency: input gradient of the DV24 head w.r.t. each ATE-signed
    treatment must match the ATE sign (A1: create_graph=True so the term has
    a live gradient w.r.t. model parameters)
  - counterfactual: for EVERY sensitivity-checked ATE treatment v,
    do(x_v = -sign(ATE_v) * 1 sigma) must decrease predicted DV24 (A7)

DV24 target (A2): regression is performed in the Yeo-Johnson transformed
space of DV24 (phase-1 dv24_yj_transformer.pkl, a sklearn PowerTransformer);
predictions are inverse-transformed back to kt for kt-space metrics.  The RI
label (DV24 >= 30 kt) is always computed from raw kt and is unaffected.

Usage (from wpac_causal_phase2/):
    KMP_DUPLICATE_LIB_OK=TRUE ~/anaconda3/python.exe -m src.train
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.dataset import RIPanelDataset
from wpac_warning.graph import load_prior_graph
from wpac_warning.models import CausalLossTerms, CausalRINet
from wpac_warning.utils_cfg import load_cfg

try:
    from torch_geometric.nn import GATv2Conv  # noqa: F401
    HAS_PYG = True
except Exception:
    HAS_PYG = False


def seeded_shuffle_generator(seed: int) -> torch.Generator:
    """Run-local DataLoader RNG, independent of model-construction RNG use."""
    return torch.Generator().manual_seed(int(seed))


def make_optimizer(model, cfg, lr: float) -> torch.optim.Optimizer:
    """AdamW with a separate, smaller lr for the ViT backbone
    (train.backbone_lr_scale; the fresh projection head and all other
    branches use the main lr).  Applies to the vit_mae and lingbot backends.
    Grouping is attribute-based, i.e. safe through TemporalAggregatorEncoder
    (whose names are prefixed "base."/"t_": a name-prefix test would misroute
    the proj head and the temporal aggregator into the backbone group).
    For lingbot, train.llrd_decay != 1.0 switches the backbone group to
    LLRD per-layer groups from LingBotEncoder.depth_param_groups (LoRA
    parameters live inside the backbone modules, so they are grouped too;
    model.vision.proj is a head and always uses the main lr)."""
    wd = float(cfg.get("train.weight_decay", 0.01))
    scale = float(cfg.get("train.backbone_lr_scale", 1.0))
    llrd = float(cfg.get("train.llrd_decay", 1.0))
    groups = []
    backbone_ids = set()
    backend = getattr(model, "vision_backend", None)
    vision = getattr(model, "vision", None)
    if backend == "vit_mae" and scale != 1.0:
        # attribute-based (wrapper-safe): the backbone is everything in the
        # unwrapped ViTMAEEncoder except its proj head; the temporal
        # aggregator (t_*) and proj fall into the main-lr rest group
        v_base = getattr(vision, "base", vision)
        head_ids = {id(p) for p in v_base.proj.parameters()}
        bp = [p for p in v_base.parameters()
              if p.requires_grad and id(p) not in head_ids]
        backbone_ids = {id(p) for p in bp}
        groups.append({"params": bp, "lr": lr * scale})
    elif backend == "lingbot" and vision is not None:
        if llrd != 1.0 and hasattr(vision, "depth_param_groups"):
            for g in vision.depth_param_groups(lr * scale, llrd):
                backbone_ids.update(id(p) for p in g["params"])
                groups.append(g)
        elif scale != 1.0:
            bp = [p for p in vision.backbone.parameters() if p.requires_grad]
            backbone_ids = {id(p) for p in bp}
            groups.append({"params": bp, "lr": lr * scale})
    rest = [p for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_ids]
    groups.append({"params": rest, "lr": lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)


def focal_loss(logit: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
               alpha: float | None = None) -> torch.Tensor:
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = torch.where(target > 0.5, p, 1 - p)
    loss = (1 - pt) ** gamma * ce
    if alpha is not None:
        # class-balance weight (0.25 = standard for rare positives; curbs
        # over-predicting the ~11% RI class)
        loss = torch.where(target > 0.5, alpha, 1.0 - alpha) * loss
    return loss.mean()


class DV24Transformer:
    """Yeo-Johnson transform of the DV24 regression target (A2).

    Loads the phase-1 transformer (sklearn PowerTransformer,
    method='yeo-johnson', standardize=False).

    In the holdout protocol (``required=True``) the path must be explicit,
    the file must load, and no refit fallback exists: a missing/corrupt
    transformer is a hard error, and an optional SHA-256 is verified before
    loading.  The legacy LOYO path keeps the historical refit fallback.
    """

    def __init__(self, cfg, path_override: Optional[str] = None,
                 required: bool = False,
                 expected_sha256: Optional[str] = None):
        self.pt = None
        self.source_path = None
        self.required = required
        path = path_override or cfg.get("data.dv24_transformer")
        if not path:
            if required:
                raise ValueError("data.dv24_transformer is required")
            path = None
        if path:
            from wpac_warning.protocol import verify_sha256
            if expected_sha256:
                verify_sha256(path, expected_sha256, "DV24 transformer")
            try:
                pt = joblib.load(path)
                # sanity: must provide transform / inverse_transform
                pt.inverse_transform(pt.transform(np.zeros((1, 1))))
                self.pt = pt
                self.source_path = str(path)
                logger.info(f"DV24 Yeo-Johnson transformer loaded from {path} "
                            f"(lambda={getattr(pt, 'lambdas_', None)})")
            except Exception as exc:  # unreadable / unexpected interface
                if required:
                    raise RuntimeError(
                        f"DV24 transformer failed to load ({exc}); refusing to "
                        f"refit under the holdout protocol") from exc
                logger.warning(f"DV24 transformer load failed ({exc}); refitting")
        if self.pt is None:
            if required:
                raise RuntimeError("no DV24 transformer available and refitting "
                                   "is disabled under the holdout protocol")
            from sklearn.preprocessing import PowerTransformer
            panel = pd.read_parquet(cfg.get("data.panel_parquet"),
                                    columns=["ISO_TIME", "DV24"])
            yr = pd.to_datetime(panel["ISO_TIME"]).dt.year
            tr = cfg.get("data.train_years", [2000, 2018])
            y = panel.loc[(yr >= tr[0]) & (yr <= tr[1]), "DV24"].values.reshape(-1, 1)
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(y)
            out = P2_ROOT / "data" / "dv24_yj_transformer_refit.pkl"
            joblib.dump(self.pt, out)
            logger.warning(f"refit DV24 transformer saved to {out}")


    def transform(self, y_kt: np.ndarray) -> np.ndarray:
        return self.pt.transform(np.asarray(y_kt, dtype=np.float64).reshape(-1, 1)).ravel()

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        return self.pt.inverse_transform(np.asarray(y_t, dtype=np.float64).reshape(-1, 1)).ravel()


def load_ate_signs(cfg) -> Dict[str, float]:
    """LEGACY LOYO ONLY: sensitivity-checked ATE signs.

    There is deliberately no fallback to ``phase1/data/processed``: a run
    that does not declare its ATE table must fail instead of silently using
    an older all-sample file.  The holdout path never calls this function; it
    reads the hash-pinned prior bundle.
    """
    ate_path = cfg.get("data.ate_csv")
    if not ate_path:
        raise ValueError("data.ate_csv is required (no legacy ATE fallback)")
    ate = pd.read_csv(ate_path)
    dv = ate[ate["outcome"] == "DV24"]
    # only trust signs that are bootstrap-significant and refutation-supported
    ok = dv[(dv["p_value_boot"] < 0.05) & (dv["refutation_passed"] >= 2)]
    return {r["treatment"]: float(np.sign(r["ate"])) for _, r in ok.iterrows()}


def run_epoch(model, loader, prior, closs, opt, device, train: bool,
              focal_gamma: float, task: str, dv24_tf: DV24Transformer,
              causal_every: int = 1, causal_scale: float = 1.0,
              focal_alpha: float | None = None) -> Dict[str, float]:
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    metrics = {"mse": 0.0, "main": 0.0, "mae_kt": 0.0, "gate": 0.0}
    ei = prior["edge_index"].to(device)
    ea = prior["edge_attr"].to(device)
    env_vars = prior["env_vars"]
    # A7: all trusted ATE treatments that are model inputs
    cf_vars = [v for v in closs.ate_signs if v in env_vars]
    for bidx, batch in enumerate(loader):
        x_env = batch["x_env"].to(device).float()
        # vision input: raw frames (B,1,H,W) / (B,T,1,H,W), or cached frozen
        # backbone embeddings (B,T,D) when data.ir_embeds_dir is set — the
        # temporal wrapper dispatches on input rank
        x_ir = batch.get("x_ir")
        if x_ir is not None:
            x_ir = x_ir.to(device).float()
        else:
            x_ir = batch["x_ir_embed"].to(device).float()
        y_dv = batch["y_dv24"].to(device).float()
        y_ri = batch["y_ri"].to(device).float()
        # multi-frame temporal input (data.ir_lag_steps): x_ir is (B,T,1,H,W)
        # and ir_mask (B,T) marks real vs t0-filled frames; absent = single-frame
        ir_mask = batch.get("ir_mask")
        if ir_mask is not None:
            ir_mask = ir_mask.to(device).float()
        env_mask = batch.get("env_mask")
        if env_mask is not None:
            env_mask = env_mask.to(device).float()

        # Causal regularizers are expensive (input gradients with
        # create_graph=True + a batched counterfactual forward); they can be
        # subsampled to every causal_every-th batch (default: every batch).
        do_causal = train and (bidx % causal_every == 0)
        with torch.set_grad_enabled(train):
            if train and do_causal:
                x_env.requires_grad_(True)
            out = model(x_env, x_ir, ei, ea, ir_mask=ir_mask,
                        env_mask=env_mask)

            # main loss
            if task == "dv24":
                y_t = torch.from_numpy(
                    dv24_tf.transform(y_dv.detach().cpu().numpy())).to(device).float()
                main = F.mse_loss(out["dv24"], y_t)
            else:
                main = focal_loss(out["ri_logit"], y_ri, gamma=focal_gamma,
                                  alpha=focal_alpha)

            if do_causal:
                # --- sign consistency (A1: create_graph=True keeps the term
                # differentiable w.r.t. model parameters) ---
                grads = {}
                if closs.w_sign > 0 and x_env.requires_grad:
                    g = torch.autograd.grad(out["dv24"].mean(), x_env,
                                            create_graph=True, retain_graph=True,
                                            allow_unused=True)[0]
                    if g is not None:
                        # per-treatment mean gradient over (batch, hist)
                        for j, v in enumerate(env_vars):
                            if v in closs.ate_signs:
                                grads[v] = g[:, :, j].mean()
                # --- counterfactual do-interventions (A7), batched over all
                # trusted treatments: do(x_v = -sign(ATE_v) * 1 sigma) should
                # decrease predicted DV24; baseline is detached so the loss
                # trains the cf response. ---
                deltas: Dict[str, torch.Tensor] = {}
                if closs.w_cf > 0 and cf_vars:
                    B = x_env.shape[0]
                    nv = len(cf_vars)
                    x_rep = x_env.detach().unsqueeze(0).repeat(nv, 1, 1, 1)
                    for k, v in enumerate(cf_vars):
                        j = env_vars.index(v)
                        x_rep[k, :, :, j] = x_rep[k, :, :, j] - float(closs.ate_signs[v])
                    x_cf = x_rep.view(nv * B, *x_env.shape[1:])
                    # the vision input is IDENTICAL across intervention
                    # variants: run the (expensive) vision branch once and
                    # reuse its embedding — gradients still flow to the
                    # vision branch through the repeated tensor.
                    env_mask_cf = (env_mask.repeat(nv, 1)
                                   if env_mask is not None else None)
                    out_cf = model(x_cf, x_ir, ei, ea,
                                   z_ir=out["z_ir"].repeat(nv, 1),
                                   env_mask=env_mask_cf)
                    for k, v in enumerate(cf_vars):
                        deltas[v] = out_cf["dv24"][k * B:(k + 1) * B] - out["dv24"].detach()
                cl = closs(grads, out["attn_edges"], deltas)
                loss = main + causal_scale * cl["total"]
            else:
                loss = main

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        bs = len(y_dv)
        tot += float(loss.detach().cpu()) * bs
        metrics["main"] += float(main.detach().cpu()) * bs
        if task == "dv24":
            metrics["mse"] += float(F.mse_loss(out["dv24"].detach(), y_t).cpu()) * bs
            pred_kt = dv24_tf.inverse(out["dv24"].detach().cpu().numpy())
            metrics["mae_kt"] += float(np.abs(pred_kt - y_dv.cpu().numpy()).mean()) * bs
        if out.get("gate") is not None:
            metrics["gate"] += float(out["gate"].detach().mean().cpu()) * bs
        n += bs
    return {"loss": tot / max(n, 1), "main": metrics["main"] / max(n, 1),
            "mse": metrics["mse"] / max(n, 1), "mae_kt": metrics["mae_kt"] / max(n, 1),
            "gate": metrics["gate"] / max(n, 1)}


@torch.no_grad()
def predict_probs(model, loader, prior, device):
    """RI probabilities + labels for a loader (eval mode, no grads)."""
    p, y, _, _, _ = predict_probs_gates(model, loader, prior, device)
    return p, y


@torch.no_grad()
def predict_probs_gates(model, loader, prior, device):
    """predict_probs plus per-row causal-branch participation diagnostics.

    gate_mean / gate_std: mean and std over the d_model fusion-gate vector
    (how strongly the fused residual blends the [env, causal] attention
    output into the vision stream).  causal_share: head-averaged cross-
    attention weight placed on the causal GNN token (0 when the GNN branch
    is disabled).  All arrays are row-aligned with the loader order.
    """
    model.eval()
    ei = prior["edge_index"].to(device)
    ea = prior["edge_attr"].to(device)
    probs, ys, gmean, gstd, cshare = [], [], [], [], []
    for batch in loader:
        x_ir = batch.get("x_ir")
        if x_ir is not None:
            x_ir = x_ir.to(device).float()
        else:
            x_ir = batch["x_ir_embed"].to(device).float()
        ir_mask = batch.get("ir_mask")
        if ir_mask is not None:
            ir_mask = ir_mask.to(device).float()
        env_mask = batch.get("env_mask")
        if env_mask is not None:
            env_mask = env_mask.to(device).float()
        out = model(batch["x_env"].to(device).float(), x_ir, ei, ea,
                    ir_mask=ir_mask, env_mask=env_mask)
        probs.append(torch.sigmoid(out["ri_logit"]).cpu().numpy())
        ys.append(batch["y_ri"].numpy())
        g = out["gate"].detach().cpu()
        gmean.append(g.mean(dim=-1).numpy())
        gstd.append(g.std(dim=-1).numpy())
        cshare.append(out["causal_attn_share"].detach().cpu().numpy())
    return (np.concatenate(probs), np.concatenate(ys),
            np.concatenate(gmean).astype(np.float32),
            np.concatenate(gstd).astype(np.float32),
            np.concatenate(cshare).astype(np.float32))


def bss_score(p: np.ndarray, y: np.ndarray) -> float:
    """Brier Skill Score vs the sample climatology."""
    bs = float(np.mean((p - y) ** 2))
    bs_clim = float(np.mean((y.mean() - y) ** 2))
    return 1.0 - bs / max(bs_clim, 1e-9)


def pick_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Decision threshold maximizing F1 on the inner-validation split."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = p >= t
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def safe_auc(p: np.ndarray, y: np.ndarray) -> tuple:
    """(ROC-AUC, PR-AUC); NaN when the held year has a single class."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(y, p)),
            float(average_precision_score(y, p)))


def _resolved(path_like) -> Path:
    p = Path(path_like)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def _check_same_path(config_path, bundle_path, what: str) -> None:
    """A config path, when present, must point at the bundle's file."""
    if not config_path:
        return
    a, b = _resolved(config_path), _resolved(bundle_path)
    if a != b:
        raise ValueError(
            f"{what}: config points at {a} but the prior manifest pins {b}. "
            f"Remove the config path or point it at the manifest file — "
            f"falling back to an older all-sample asset is not allowed.")


def _hash_optional(path: Optional[str], what: str) -> Optional[Dict[str, str]]:
    if not path:
        return None
    p = _resolved(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found: {p}")
    from wpac_warning.protocol import sha256_file
    return {"path": str(p), "sha256": sha256_file(p)}


def verify_canonical_sample_manifest(cfg, bundle) -> dict:
    """Recompute the config's canonical SID/year content hash against the prior."""
    from wpac_warning.protocol import load_storm_year_manifest, sha256_sid_years
    path_value = cfg.get("data.storm_year_manifest")
    if not path_value:
        raise ValueError("data.storm_year_manifest is required")
    path = _resolved(path_value)
    mapping = load_storm_year_manifest(path)
    actual = sha256_sid_years(mapping.to_dict())
    expected = (bundle.get("population") or {}).get(
        "canonical_sample_content_sha256")
    if not expected or actual != expected:
        raise ValueError(
            "config storm_year_manifest canonical SID/year hash differs from "
            "the prior population record")
    return {"path": str(path), "canonical_sample_content_sha256": actual}


def run_holdout(cfg, resume: bool = False) -> int:
    """HOLDOUT_MASKED_V1 training: fit -> inner early stopping, inner-only.

    No test loader is built and no test prediction is written: test inference
    is a separate step that requires a frozen ensemble protocol lock.
    """
    from wpac_warning.checkpoint import (backbone_digest, backbone_fully_frozen,
                                save_inference_checkpoint)
    from wpac_warning.dataset import RIPanelDataset
    from wpac_warning.graph import load_prior_graph, shuffle_prior_graph
    from wpac_warning.models import CausalLossTerms, CausalRINet
    from wpac_warning.protocol import (RunDir, SplitProtocol, load_prior_bundle,
                              load_trusted_signs, sha256_file, sha256_json)
    from wpac_warning.utils_cfg import Config

    proto = SplitProtocol.from_cfg(cfg)
    proto.validate()
    logger.info(f"protocol: {proto.as_dict()}")

    # ---- required prior bundle (hash-verified, no fallback) ------------
    prior_manifest = cfg.get("data.prior_manifest")
    if not prior_manifest:
        raise ValueError("data.prior_manifest is required under the holdout "
                         "protocol (there is no legacy ATE fallback)")
    bundle = load_prior_bundle(_resolved(prior_manifest), proto)
    canonical_sample = verify_canonical_sample_manifest(cfg, bundle)
    for key, cfg_key, what in (
            ("panel", "data.panel_parquet", "panel"),
            ("graph_edges", "data.graph_edges_json", "prior graph"),
            ("ate", "data.ate_csv", "ATE table"),
            ("identification", "data.identification_csv", "identification table")):
        _check_same_path(cfg.get(cfg_key), bundle["files"][key]["path"], what)
    if "dv24_transformer" not in bundle["assets"]:
        raise ValueError("prior manifest must pin assets.dv24_transformer")
    if "tbb_norm" not in bundle["assets"]:
        raise ValueError("prior manifest must pin assets.tbb_norm (fit-only)")
    _check_same_path(cfg.get("data.dv24_transformer"),
                     bundle["assets"]["dv24_transformer"]["path"],
                     "DV24 transformer")
    _check_same_path(cfg.get("data.tbb_norm"),
                     bundle["assets"]["tbb_norm"]["path"],
                     "TBB normalisation")
    if "frames_index" in bundle["assets"]:
        _check_same_path(cfg.get("data.frames_index"),
                         bundle["assets"]["frames_index"]["path"],
                         "frames index")
    if "frames_qc" not in bundle or "frame_availability" not in bundle:
        raise ValueError("prior bundle must pin frames_qc and frame_availability")
    _check_same_path(cfg.get("data.frames_dir"),
                     bundle["frames_qc"]["out_frames_dir"],
                     "frames directory")
    _check_same_path(cfg.get("data.frame_availability"),
                     bundle["frame_availability"]["path"],
                     "frame availability")
    logger.info(f"prior manifest verified: {bundle['manifest_path']} "
                f"(sha256={bundle['manifest_sha256'][:16]}…, "
                f"{len(bundle['sids'])} SIDs, years {bundle['years'][0]}-"
                f"{bundle['years'][-1]}, mask={bundle['mask_type']})")

    resolved = copy.deepcopy(cfg.to_dict())
    resolved["data"]["graph_edges_json"] = bundle["files"]["graph_edges"]["path"]
    resolved["data"]["panel_parquet"] = bundle["files"]["panel"]["path"]
    resolved["data"]["ate_csv"] = bundle["files"]["ate"]["path"]
    resolved["data"]["identification_csv"] = bundle["files"]["identification"]["path"]
    resolved["data"]["dv24_transformer"] = bundle["assets"]["dv24_transformer"]["path"]
    resolved["data"]["tbb_norm"] = bundle["assets"]["tbb_norm"]["path"]
    resolved["data"]["frames_dir"] = bundle["frames_qc"]["out_frames_dir"]
    resolved["data"]["frame_availability"] = bundle["frame_availability"]["path"]
    if "frames_index" in bundle["assets"]:
        resolved["data"]["frames_index"] = bundle["assets"]["frames_index"]["path"]
    graph_cfg = Config(resolved)
    prior = load_prior_graph(
        graph_cfg,
        expected_sha256=bundle["files"]["graph_edges"]["sha256"],
        require_metadata=True)
    shuffle_seed = cfg.get("model.gnn.prior_shuffle_seed", None)
    if shuffle_seed is not None and int(shuffle_seed) > 0:
        prior = shuffle_prior_graph(prior, int(shuffle_seed))
        logger.warning(f"C3 ABLATION: prior graph shuffled "
                       f"(seed={int(shuffle_seed)}) on the fold prior "
                       f"{prior.get('source_sha256')}")
    logger.info(f"prior graph: {prior['edge_index'].shape[1]} edges "
                f"({prior['n_env']} env nodes + readout; "
                f"dropped weak: {prior['n_dropped_weak']})")

    # ---- trusted signs: fail-closed -------------------------------------
    exclude = list(cfg.get("data.ate_exclude_treatments", ["D200"]) or [])
    signs, sign_audit = load_trusted_signs(
        bundle["files"]["ate"]["path"],
        bundle["files"]["identification"]["path"],
        exclude=exclude,
        outcome="DV24",
        p_boot_max=float(cfg.get("data.ate_p_boot_max", 0.05)),
        min_refutations=int(cfg.get("data.ate_min_refutations", 2)))
    logger.info(f"ATE signs trusted (fail-closed): {signs}")
    for row in sign_audit:
        if not row["trusted"]:
            logger.info(f"  excluded {row['treatment']}: {row['reason']}")

    dv24_tf = DV24Transformer(
        cfg,
        path_override=bundle["assets"]["dv24_transformer"]["path"],
        required=True,
        expected_sha256=bundle["assets"]["dv24_transformer"]["sha256"])

    # ---- data + split ---------------------------------------------------
    ds = RIPanelDataset(Config(resolved))
    if (resolved["data"].get("ir_embeds_dir")
            and resolved["data"].get("ir_lag_steps")
            and ds.embedding_cache_audit is None):
        raise ValueError("embedding cache was used without a provenance audit")
    splits = ds.split_indices
    for split in ("fit", "inner"):
        if len(splits[split]) == 0:
            raise ValueError(f"empty '{split}' split after filtering")
    if len(splits["test"]) == 0:
        logger.warning("test split is empty after filtering — it will produce "
                       "no test predictions, which is expected at this stage")
    n_pos_inner = int(ds.df.iloc[splits["inner"]]["RI"].sum())
    if n_pos_inner == 0:
        raise ValueError("inner split has zero RI positives — early stopping "
                         "and threshold selection would be undefined")
    logger.info("split rows: fit={} inner={} test={} (storms {} / {} / {})".format(
        len(splits["fit"]), len(splits["inner"]), len(splits["test"]),
        len(ds.split_manifest["splits"]["fit"]["sids"]),
        len(ds.split_manifest["splits"]["inner"]["sids"]),
        len(ds.split_manifest["splits"]["test"]["sids"])))

    # ---- versioned run directory + manifests ---------------------------
    run_root = _resolved(cfg.get("train.output_root", "data/runs"))
    run_id = str(cfg.get("train.run_id", "") or "")
    if not run_id:
        raise ValueError("train.run_id is required (versioned output root)")
    data_manifest = {
        "protocol": proto.as_dict(),
        "panel": _hash_optional(resolved["data"]["panel_parquet"], "panel"),
        "frames_index": _hash_optional(resolved["data"]["frames_index"],
                                       "frames index"),
        "storm_year_manifest": _hash_optional(
            cfg.get("data.storm_year_manifest"), "sample manifest"),
        "canonical_sample": canonical_sample,
        "tbb_norm": _hash_optional(resolved["data"]["tbb_norm"], "TBB norm"),
        "frames_dir": str(_resolved(cfg.get("data.frames_dir"))),
        "frames_qc": ds.frames_qc_audit,
        "frame_availability": ds.frame_availability_audit,
        "ir_embeds_dir": str(cfg.get("data.ir_embeds_dir", "") or ""),
        "ir_embeds": ds.embedding_cache_audit,
        "identification_dag": bundle.get("identification_dag"),
        "smoke_limit_rows": int(ds.smoke_limit_rows),
        "n_rows": int(len(ds.df)),
        "n_rows_by_split": {k: int(len(v)) for k, v in splits.items()},
    }
    seed = int(cfg.get("train.seed", 42))
    config_sha = sha256_json(cfg.to_dict())
    payload = {
        "protocol": proto.as_dict(),
        "config_sha256": config_sha,
        "prior_manifest_sha256": bundle["manifest_sha256"],
        "prior_files": {k: v["sha256"] for k, v in bundle["files"].items()},
        "frame_availability": ds.frame_availability_audit,
        "data_manifest": data_manifest,
        "seed": seed,
        "run_id": run_id,
        "inner_only": True,
        "created_by": "src.train --protocol holdout_masked_v1",
    }
    run = RunDir(run_root, run_id, resume=resume)
    run.prepare(payload)
    logger.add(str(run.path / "logs" / "run.log"), level="INFO")
    logger.info(f"[{run_id}] config_sha256={config_sha[:16]}… "
                f"prior_manifest_sha256={bundle['manifest_sha256'][:16]}…")
    run.write_config(cfg.to_dict())
    run.write_json("protocol.json", proto.as_dict())
    run.write_json("prior_manifest.json", bundle)
    run.write_json("data_manifest.json", data_manifest)
    run.write_json("split_manifest.json", ds.split_manifest)
    run.write_json("ate_sign_audit.json", {"signs": signs, "rows": sign_audit})
    results_path = run.path / "results.csv"
    ckpt_path = run.path / "checkpoints" / "best_ri.pt"
    inner_npz = run.path / "preds" / "inner.npz"
    if resume and (run.path / "completed.json").is_file():
        from wpac_warning.run_integrity import load_completed_marker
        load_completed_marker(run.path)
        logger.info(f"[{run_id}] completed marker and artifact hashes verified "
                    f"— explicit --resume: skipping (test is never touched)")
        return 0
    if not bool(cfg.get("train.inner_only", True)):
        raise ValueError(
            "the holdout protocol forbids test predictions during training; "
            "keep train.inner_only=true and use scripts/test_infer.py after "
            "the protocol lock is frozen")
    if resume:
        logger.info(f"[{run_id}] explicit --resume with incomplete outputs: "
                    f"retraining from epoch 0 inside the same "
                    f"fingerprint-verified run directory")

    # ---- loaders, model, losses ----------------------------------------
    bs = int(cfg.get("train.batch_size", 16))
    nw = int(cfg.get("train.num_workers", 0))
    pw = nw > 0
    fit_subset = torch.utils.data.Subset(ds, splits["fit"])
    tr_loader = DataLoader(fit_subset, batch_size=bs, shuffle=True,
                           num_workers=nw, persistent_workers=pw,
                           generator=seeded_shuffle_generator(seed))
    first_fit_order = list(torch.utils.data.RandomSampler(
        fit_subset, generator=seeded_shuffle_generator(seed)))
    shuffle_audit = {"seed": seed, "batch_size": bs,
                     "n_rows": len(first_fit_order),
                     "first_epoch_order_sha256": sha256_json(first_fit_order)}
    in_loader = DataLoader(torch.utils.data.Subset(ds, splits["inner"]),
                           batch_size=bs, shuffle=False, num_workers=nw,
                           persistent_workers=pw)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    from wpac_warning.mae_provenance import verify_mae_for_config
    mae_provenance = verify_mae_for_config(cfg)
    model = CausalRINet(cfg, prior=prior).to(device)

    from wpac_warning.run_integrity import (load_initial_checkpoint,
                                   save_initial_checkpoint)
    init_from = cfg.get("train.init_from")
    load_audit = None
    if init_from:
        load_audit = load_initial_checkpoint(
            model, _resolved(init_from),
            allow_structural=not bool(cfg.get("model.gnn.enabled", True)))
    initial = save_initial_checkpoint(
        run.path / "checkpoints" / "initial.pt", model,
        run_id=run_id, seed=seed, config_sha256=config_sha, source=load_audit)
    run.write_json("initialization.json", {
        "seed": seed,
        "shuffle_generator": {"type": "torch.Generator", "seed": seed,
                              "independent_of_model_rng": True},
        "fit_shuffle": shuffle_audit,
        "initial_checkpoint": {"path": initial["path"],
                               "sha256": initial["sha256"]},
        "source": initial["meta"]["source"],
        "load_audit": load_audit,
    })
    # Model construction and C2 structural merging must not shift training RNG.
    torch.manual_seed(seed)
    np.random.seed(seed)

    closs = CausalLossTerms(cfg, prior, signs).to(device)
    weight_manifest = {
        "backend": str(getattr(model, "vision_backend", "")),
        "vision_config": dict(cfg.get("model.vision", {}) or {}),
        "trainable_params": int(model.count_params()),
        "backbone_fully_frozen": bool(backbone_fully_frozen(model)),
        "backbone_sha256": backbone_digest(model),
        "mae_checkpoint": _hash_optional(cfg.get("model.vision.mae_checkpoint"),
                                         "MAE checkpoint"),
        "mae_provenance": mae_provenance,
        "initialization_sha256": initial["sha256"],
        "lingbot_dir": str(cfg.get("model.vision.lingbot_dir", "") or ""),
    }
    if weight_manifest["lingbot_dir"]:
        weight_manifest["lingbot_model"] = _hash_optional(
            Path(weight_manifest["lingbot_dir"]) / "model.pt", "lingbot model.pt")
    run.write_json("weight_manifest.json", weight_manifest)
    logger.info(f"[{run_id}] trainable params: {model.count_params():,}; "
                f"backbone frozen={weight_manifest['backbone_fully_frozen']} "
                f"hash={weight_manifest['backbone_sha256'][:16]}…")

    causal_every = int(cfg.get("train.causal_every", 1))
    causal_ft_scale = float(cfg.get("train.causal_finetune_scale", 1.0))
    fg = float(cfg.get("train.focal_gamma", 2.0))
    fa = float(cfg.get("train.focal_alpha", 0.25))
    patience = int(cfg.get("train.early_stop_patience", 10))

    # ---- phase A: DV24 pretrain, early stop on inner MSE ---------------
    best_mse, best_state = float("inf"), None
    stale = 0
    n_pre = int(cfg.get("train.pretrain_epochs", 30))
    opt = make_optimizer(model, cfg, float(cfg.get("train.lr_pretrain", 1e-3)))
    for ep in range(n_pre):
        tr = run_epoch(model, tr_loader, prior, closs, opt, device, True, fg,
                       task="dv24", dv24_tf=dv24_tf, causal_every=causal_every,
                       focal_alpha=fa)
        va_in = run_epoch(model, in_loader, prior, closs, opt, device, False, fg,
                          task="dv24", dv24_tf=dv24_tf)
        if va_in["mse"] < best_mse:
            best_mse, best_state = va_in["mse"], copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if ep % 5 == 0 or ep == n_pre - 1:
            logger.info(f"[{run_id}] pretrain ep{ep}: tr_loss={tr['loss']:.4f} "
                        f"inner_mse={va_in['mse']:.3f} "
                        f"inner_mae={va_in['mae_kt']:.2f}kt "
                        f"(best {best_mse:.3f})")
        if stale >= patience:
            logger.info(f"[{run_id}] pretrain early stop at ep{ep}")
            break
    if best_state is None:
        raise RuntimeError("pretrain produced no checkpoint")
    model.load_state_dict(best_state)

    # ---- phase B: RI finetune, early stop on inner BSS -----------------
    lr_ft = float(cfg.get("train.lr_finetune", 3e-4))
    lpft_k = int(cfg.get("train.lpft_warmup_epochs", 0))
    lpft = (lpft_k > 0
            and getattr(model, "vision_backend", None) == "lingbot"
            and hasattr(model.vision, "set_frozen_override"))
    if lpft:
        model.vision.set_frozen_override(True)
    opt = make_optimizer(model, cfg, lr_ft)
    best_bss, best_state, best_ep = -float("inf"), None, -1
    stale = 0
    n_ft = int(cfg.get("train.finetune_epochs", 40))
    for ep in range(n_ft):
        if lpft and ep == lpft_k:
            model.vision.set_frozen_override(False)
            opt = make_optimizer(model, cfg, lr_ft)
        tr = run_epoch(model, tr_loader, prior, closs, opt, device, True, fg,
                       task="ri", dv24_tf=dv24_tf, causal_every=causal_every,
                       causal_scale=causal_ft_scale, focal_alpha=fa)
        p_in, y_in = predict_probs(model, in_loader, prior, device)
        b_in = bss_score(p_in, y_in)
        if b_in > best_bss:
            best_bss, best_state, best_ep = b_in, copy.deepcopy(
                model.state_dict()), ep
            stale = 0
        else:
            stale += 1
        if ep % 5 == 0 or ep == n_ft - 1:
            logger.info(f"[{run_id}] finetune ep{ep}: tr_loss={tr['loss']:.4f} "
                        f"inner_bss={b_in:.4f} (best {best_bss:.4f})")
        if stale >= patience:
            logger.info(f"[{run_id}] finetune early stop at ep{ep}")
            break
    if best_state is None:
        raise RuntimeError("finetune produced no checkpoint")
    model.load_state_dict(best_state)

    # ---- threshold on inner, checkpoint + inner predictions ------------
    p_in, y_in = predict_probs(model, in_loader, prior, device)
    thr = pick_threshold(p_in, y_in)
    bss = bss_score(p_in, y_in)
    inner_rows = ds.df.iloc[splits["inner"]].reset_index(drop=True)
    if len(inner_rows) != len(p_in):
        raise RuntimeError(
            f"inner prediction/row-key length mismatch: {len(p_in)} vs "
            f"{len(inner_rows)} rows")
    key_arrays = {
        "sid": inner_rows["SID"].astype(str).values,
        "iso_time": inner_rows["iso_time_str"].values,
        "storm_year": inner_rows["storm_year"].astype(np.int32).values,
        "row_uid": inner_rows["row_uid"].values,
        "split": np.full(len(inner_rows), "inner", dtype=object),
    }
    np.savez(inner_npz, p_inner=p_in, y_inner=y_in, threshold=float(thr),
             inner_bss=float(bss), best_epoch=int(best_ep),
             run_id=run_id, **key_arrays)

    meta = {
        "protocol": proto.name,
        "run_id": run_id,
        "best_epoch": int(best_ep),
        "inner_bss": float(bss),
        "threshold": float(thr),
        "seed": seed,
        "config_sha256": config_sha,
        "prior_manifest_sha256": bundle["manifest_sha256"],
        "graph_sha256": bundle["files"]["graph_edges"]["sha256"],
        "ate_sha256": bundle["files"]["ate"]["sha256"],
        "identification_sha256": bundle["files"]["identification"]["sha256"],
        "ate_signs": signs,
        "prior_shuffle_seed": int(shuffle_seed) if shuffle_seed else 0,
        "inner_only": True,
    }
    payload_ckpt = save_inference_checkpoint(ckpt_path, model, meta)
    logger.info(f"[{run_id}] checkpoint saved: mode={payload_ckpt['mode']} "
                f"trainable_params={payload_ckpt['trainable_params']:,} "
                f"backbone={payload_ckpt['backbone']['sha256'][:16]}…")

    row = {
        "run_id": run_id, "policy": "inner_only",
        "n_fit": len(splits["fit"]), "n_inner": len(splits["inner"]),
        "n_test_rows_untouched": len(splits["test"]),
        "inner_bss": bss, "threshold": thr, "best_epoch": best_ep,
        "inner_ri_rate": float(y_in.mean()),
        "prior_manifest_sha256": bundle["manifest_sha256"],
        "graph_sha256": bundle["files"]["graph_edges"]["sha256"],
        "ate_sha256": bundle["files"]["ate"]["sha256"],
        "identification_sha256": bundle["files"]["identification"]["sha256"],
        "ate_signs": ";".join(f"{k}{int(v):+d}" for k, v in sorted(signs.items())),
        "checkpoint": str(ckpt_path.relative_to(run.path)),
        "inner_npz": str(inner_npz.relative_to(run.path)),
        "trainable_params": payload_ckpt["trainable_params"],
        "ckpt_mode": payload_ckpt["mode"],
    }
    pd.DataFrame([row]).to_csv(results_path, index=False)
    from wpac_warning.run_integrity import write_completed_marker
    write_completed_marker(run.path, run_id)
    logger.info(f"[{run_id}] inner-only done: BSS={bss:.4f} thr={thr:.3f}; "
                f"completed marker atomically written → {results_path}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "phase2" / "config.yaml"))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="dot-path config override, e.g. --set model.gnn.enabled=false "
                         "(repeatable; value parsed as YAML)")
    ap.add_argument("--resume", action="store_true",
                    help="explicitly continue an existing run directory (same "
                         "fingerprint); implicit fold skipping is disabled")
    ap.add_argument("--log-file", default=str(P2_ROOT / "logs" / "train.log"),
                    help="process-level log path (parallel runners use per-arm files)")
    args = ap.parse_args()

    logger.add(args.log_file, level="INFO", rotation="10 MB")
    cfg = load_cfg(args.config)
    for kv in args.overrides:
        key, _, raw = kv.partition("=")
        val = yaml.safe_load(raw)
        node = cfg.to_dict()
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
        logger.info(f"override: {key} = {val!r}")

    base_seed = int(cfg.get("train.seed", 42))
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    protocol = str(cfg.get("data.protocol", "loyo"))
    if protocol == "holdout_masked_v1":
        return run_holdout(cfg, resume=bool(args.resume))
    if protocol != "loyo":
        raise ValueError(f"unknown data.protocol {protocol!r}")

    prior = load_prior_graph(cfg)
    _shuffle_seed = cfg.get("model.gnn.prior_shuffle_seed", None)
    if _shuffle_seed is not None and int(_shuffle_seed) > 0:
        from wpac_warning.graph import shuffle_prior_graph
        prior = shuffle_prior_graph(prior, int(_shuffle_seed))
        logger.warning(f"C3 ABLATION: prior graph shuffled (degree-preserving, "
                       f"seed={int(_shuffle_seed)})")
    logger.info(f"prior graph: {prior['edge_index'].shape[1]} edges "
                f"({prior['n_env']} env nodes + 1 target readout node; "
                f"dropped weak p>{cfg.get('data.prior_min_pvalue', 0.05)}: "
                f"{prior['n_dropped_weak']}, dropped non-env: {prior['n_dropped_node']})")
    ate_signs = load_ate_signs(cfg)
    logger.info(f"ATE signs trusted: {ate_signs}")
    dv24_tf = DV24Transformer(cfg)

    results_path = P2_ROOT / str(cfg.get("train.results_csv", "data/loyo_results.csv"))
    loyo_years = list(cfg.get("train.loyo_years", [2019, 2020, 2021, 2022, 2023, 2024]))
    # dataset is fold-independent; build once (was rebuilt per fold)
    train_ds = RIPanelDataset(cfg)
    train_df = train_ds.df
    # LOYO split BY STORM: a storm belongs to the calendar year of its first
    # time step, so cross-year storms never straddle the train/val boundary
    sid_year = train_df.groupby("SID")["ISO_TIME"].min().dt.year
    causal_every = int(cfg.get("train.causal_every", 1))
    causal_ft_scale = float(cfg.get("train.causal_finetune_scale", 1.0))
    nw = int(cfg.get("train.num_workers", 0))
    # explicit resume only: a pre-existing results CSV aborts unless --resume
    done_years = set()
    if results_path.exists():
        done_df = pd.read_csv(results_path)
        if not args.resume:
            raise RuntimeError(
                f"{results_path} already exists — pass --resume to continue it "
                f"(implicit fold skipping is disabled)")
        full = ~done_df.get("inner_only", pd.Series(False, index=done_df.index)).fillna(False).astype(bool)
        done_years = set(done_df.loc[full, "held_year"].astype(int))
        if done_years:
            logger.info(f"explicit resume: folds already done: {sorted(done_years)}")
    results = []
    for held in loyo_years:
        if held in done_years:
            logger.info(f"=== LOYO fold {held}: already done, skipping ===")
            continue
        # anchor RNG per fold: resume and fresh runs see identical streams
        torch.manual_seed(base_seed + held)
        np.random.seed((base_seed + held) % (2**32 - 1))
        logger.info(f"=== LOYO fold: held-out year {held} ===")
        val_sids = sid_year[sid_year == held].index
        val_mask = train_df["SID"].isin(val_sids).values
        if val_mask.sum() == 0:
            continue
        # inner validation = the THREE most recent training years (storm-level
        # disjoint).  Used ONLY for early stopping / threshold selection; the
        # held-out year is touched once, for the final metric.  Restricted to
        # the training window (<= train_years[1]); multi-year inner because a
        # single inner year was found to ANTI-correlate with held-year skill
        # (inner_bss Spearman rho=-0.47 vs held BSS over 36 arm x fold points).
        train_end = int(cfg.get("data.train_years", [2000, 2018])[1])
        inner_years = [y for y in (train_end - 2, train_end - 1, train_end)
                       if y != held]
        inner_mask = train_df["SID"].isin(
            sid_year[sid_year.isin(inner_years)].index).values
        # data.train_exclude_years (optional): storm-years that may appear as
        # the held fold but NEVER in training.  Used by the RITCF final model:
        # loyo over the two test years [2020, 2021] with
        # train_exclude_years=[2020, 2021] gives both folds the identical
        # 2000-2019 training set (default [] = legacy behaviour).
        exclude_years = {int(y) for y in np.atleast_1d(
            cfg.get("data.train_exclude_years", []) or [])}
        if exclude_years:
            excl_mask = train_df["SID"].isin(
                sid_year[sid_year.isin(exclude_years)].index).values
            logger.info(f"train_exclude_years={sorted(exclude_years)}: "
                        f"{int(excl_mask.sum())} rows excluded from training")
        else:
            excl_mask = np.zeros(len(train_df), dtype=bool)
        tr_idx = np.where(~val_mask & ~inner_mask & ~excl_mask)[0]
        in_idx = np.where(inner_mask)[0]
        va_idx = np.where(val_mask)[0]
        bs = int(cfg.get("train.batch_size", 16))
        # persistent workers: avoid per-epoch respawn RAM spikes (Windows spawn
        # pickles the whole dataset per worker per epoch)
        pw = nw > 0
        tr_loader = DataLoader(torch.utils.data.Subset(train_ds, tr_idx),
                               batch_size=bs, shuffle=True, num_workers=nw,
                               persistent_workers=pw,
                               generator=seeded_shuffle_generator(base_seed + held))
        in_loader = DataLoader(torch.utils.data.Subset(train_ds, in_idx),
                               batch_size=bs, shuffle=False, num_workers=nw,
                               persistent_workers=pw)
        va_loader = DataLoader(torch.utils.data.Subset(train_ds, va_idx),
                               batch_size=bs, shuffle=False, num_workers=nw,
                               persistent_workers=pw)

        model = CausalRINet(cfg, prior=prior).to(device)
        logger.info(f"[{held}] model trainable params: {model.count_params():,}")
        closs = CausalLossTerms(cfg, prior, ate_signs).to(device)
        opt = make_optimizer(model, cfg, float(cfg.get("train.lr_pretrain", 1e-3)))
        fg = float(cfg.get("train.focal_gamma", 2.0))
        fa = float(cfg.get("train.focal_alpha", 0.25))
        logger.info(f"[{held}] inner-val years: {inner_years} "
                    f"(tr={len(tr_idx)} in={len(in_idx)} va={len(va_idx)})")

        # Phase A: DV24 regression pretrain (Yeo-Johnson target), keep best
        # inner-val MSE checkpoint; stop early after `patience` stale epochs
        best_mse, best_state = float("inf"), None
        patience = int(cfg.get("train.early_stop_patience", 10))
        stale = 0
        n_pre = int(cfg.get("train.pretrain_epochs", 30))
        for ep in range(n_pre):
            tr = run_epoch(model, tr_loader, prior, closs, opt, device, True, fg,
                           task="dv24", dv24_tf=dv24_tf, causal_every=causal_every,
                           focal_alpha=fa)
            va_in = run_epoch(model, in_loader, prior, closs, opt, device, False, fg,
                              task="dv24", dv24_tf=dv24_tf)
            if va_in["mse"] < best_mse:
                best_mse, best_state = va_in["mse"], copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if ep % 5 == 0 or ep == n_pre - 1:
                logger.info(f"[{held}] pretrain ep{ep}: tr_loss={tr['loss']:.4f} "
                            f"inner_mse={va_in['mse']:.3f} "
                            f"inner_mae={va_in['mae_kt']:.2f}kt (best {best_mse:.3f}) "
                            f"gate={tr['gate']:.3f}")
            if stale >= patience:
                logger.info(f"[{held}] pretrain early stop at ep{ep} "
                            f"(best inner_mse {best_mse:.3f})")
                break
        model.load_state_dict(best_state)

        # Phase B: RI finetune (focal) with a FRESH optimizer (no pretrain
        # momentum carry-over), keep best inner-val BSS checkpoint
        lr_ft = float(cfg.get("train.lr_finetune", 3e-4))
        # LP-FT (linear-probe then fine-tune): for the first
        # train.lpft_warmup_epochs epochs the lingbot vision backbone is
        # fully frozen (== frozen mode); at epoch K it switches back to the
        # arm's configured ft_mode and the optimizer is rebuilt (so the
        # newly trainable backbone params get their backbone_lr_scale lr).
        # Early-stopping / best-checkpoint state runs continuously across
        # the switch.
        lpft_k = int(cfg.get("train.lpft_warmup_epochs", 0))
        lpft = (lpft_k > 0
                and getattr(model, "vision_backend", None) == "lingbot"
                and hasattr(model.vision, "set_frozen_override"))
        if lpft:
            model.vision.set_frozen_override(True)
            logger.info(f"[{held}] LP-FT: vision backbone frozen for the first "
                        f"{lpft_k} finetune epochs (then ft_mode="
                        f"'{model.vision.ft_mode}')")
        opt = make_optimizer(model, cfg, lr_ft)
        best_bss, best_state = -float("inf"), None
        stale = 0
        n_ft = int(cfg.get("train.finetune_epochs", 40))
        for ep in range(n_ft):
            if lpft and ep == lpft_k:
                model.vision.set_frozen_override(False)
                opt = make_optimizer(model, cfg, lr_ft)
                logger.info(f"[{held}] LP-FT: backbone unfrozen at ep{ep} "
                            f"(ft_mode='{model.vision.ft_mode}'), optimizer rebuilt")
            tr = run_epoch(model, tr_loader, prior, closs, opt, device, True, fg,
                           task="ri", dv24_tf=dv24_tf, causal_every=causal_every,
                           causal_scale=causal_ft_scale, focal_alpha=fa)
            p_in, y_in = predict_probs(model, in_loader, prior, device)
            b_in = bss_score(p_in, y_in)
            if b_in > best_bss:
                best_bss, best_state = b_in, copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if ep % 5 == 0 or ep == n_ft - 1:
                logger.info(f"[{held}] finetune ep{ep}: tr_loss={tr['loss']:.4f} "
                            f"inner_bss={b_in:.4f} (best {best_bss:.4f}) "
                            f"gate={tr['gate']:.3f}")
            if stale >= patience:
                logger.info(f"[{held}] finetune early stop at ep{ep} "
                            f"(best inner_bss {best_bss:.4f})")
                break
        model.load_state_dict(best_state)

        # inner-only selection mode: skip thresholding + held-year eval entirely
        # (used for backbone/config selection runs; held years stay untouched)
        if bool(cfg.get("train.inner_only", False)):
            logger.info(f"[{held}] INNER-ONLY selection run: "
                        f"best inner_bss={best_bss:.4f} (held year untouched)")
            row = {"held_year": held, "inner_year": int(max(inner_years)),
                   "inner_bss": float(best_bss), "inner_only": True}
            results.append(row)
            pd.DataFrame([row]).to_csv(results_path, mode="a", index=False,
                                       header=not results_path.exists())
            continue

        # threshold on inner-val, then ONE final eval on the held-out year
        p_in, y_in = predict_probs(model, in_loader, prior, device)
        thr = pick_threshold(p_in, y_in)
        p, y = predict_probs(model, va_loader, prior, device)
        pred = p >= thr
        pod = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        far = float(((pred == 1) & (y == 0)).sum() / max((pred == 1).sum(), 1))
        bss = bss_score(p, y)
        roc_auc, pr_auc = safe_auc(p, y)
        logger.info(f"[{held}] RI: POD={pod:.3f} FAR={far:.3f} BSS={bss:.3f} "
                    f"ROC-AUC={roc_auc:.3f} PR-AUC={pr_auc:.3f} "
                    f"n={len(y)} thr={thr:.2f}")
        row = {"held_year": held, "POD": pod, "FAR": far, "BSS": bss,
               "ROC_AUC": roc_auc, "PR_AUC": pr_auc,
               "n": len(y), "ri_rate": float(y.mean()),
               "threshold": thr, "inner_year": int(max(inner_years)),
               "inner_bss": float(best_bss)}
        results.append(row)
        # optional: persist per-sample probs for ensemble/pooled analysis
        if bool(cfg.get("train.save_preds", False)):
            run_name = str(cfg.get("train.run_name", "main"))
            pred_dir = P2_ROOT / "data" / "preds"
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez(pred_dir / f"{run_name}_fold{held}.npz",
                     p_held=p, y_held=y, p_inner=p_in, y_inner=y_in)
            logger.info(f"[{held}] predictions saved (run={run_name})")
        # A4: persist per fold (append; never lose finished folds to a crash)
        pd.DataFrame([row]).to_csv(results_path, mode="a", index=False,
                                   header=not results_path.exists())

    out_df = pd.DataFrame(results)
    logger.info(f"LOYO done:\n{out_df.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    if not HAS_PYG:
        raise RuntimeError("torch_geometric required")
    sys.exit(main())
