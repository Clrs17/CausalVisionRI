# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Precompute frozen-backbone frame embeddings for the temporal-vision arms.

Encodes EVERY frame of every storm store in the configured frame corpus
(``data.frames_dir``: ``data/global_hursat/frames_qc`` for the all-TC holdout
protocol — the QC corpus stays the downstream frame source even though the
MAE weights are pretrained on the full multi-sensor corpus — the legacy WNP
``data/frames`` only for legacy runs) with the chosen vision backbone in
frozen mode and stores the POOLED BACKBONE vector (pre-proj: lingbot
x_norm_patchtokens mean -> D=384/768; vit_mae cls/mean pool -> D=384) as
data/frame_embeds/{backend_tag}/{SID}.npy, (T_i, D) float16, row-aligned
with the store's ``{SID}.npy`` / ``{SID}_iso.npy`` or ``{SID}_times.npy``
time axis.  For the ``mae_global_full_cls`` registry role the encoder is
``checkpoints/holdout_masked_v1/mae_global_full/mae_vit_s16_best.pt`` and the
manifest provenance records ``mae_corpus=hursat_full_multisensor_v1``.

Frame normalization is BYTE-IDENTICAL to the training pipeline: the call
goes through RIPanelDataset._norm_frame (reused, not copied).

Usage (from the release root):
    python scripts/03_assets/precompute_frame_embeds.py --backend lingbot \
        --dir external/lingbot-base --variant base --batch 96
    python scripts/03_assets/precompute_frame_embeds.py --backend vit_mae \
        --checkpoint checkpoints/mae_vit_s16_best.pt --pool cls
    # subset / resume / re-verify:
    ... --sids 2000125N06136,2000184N13133 --out data/frame_embeds/test
    ... --verify-only

Existing files are skipped only after full provenance and per-SID fingerprint
verification.  Holdout caches cannot be adopted without re-encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.dataset import RIPanelDataset  # noqa: E402  (for _norm_frame reuse)
from wpac_warning.embedding_cache import (  # noqa: E402
    fit_only_norm_sha256, make_provenance, resolved, validate_contract,
    write_manifest)
from wpac_warning.protocol import validate_qc_manifest  # noqa: E402
from wpac_warning.mae_provenance import verify_mae_for_config  # noqa: E402
from wpac_warning.utils_cfg import load_cfg  # noqa: E402


class _NormShim:
    """Minimal attribute holder so RIPanelDataset._norm_frame can be reused
    unbound (it only reads self.tbb_mean / self.tbb_std)."""

    def __init__(self, mean: float, std: float):
        self.tbb_mean = mean
        self.tbb_std = std


def load_norm_shim(path: str | None) -> _NormShim:
    p = Path(path) if path else P2_ROOT / "data" / "tbb_norm.json"
    norm = json.loads(p.read_text(encoding="utf-8"))
    return _NormShim(float(norm["mean"]), float(norm["std"]))


def norm_frame(shim: _NormShim, arr, j: int) -> np.ndarray:
    # THE single source of truth for frame normalization (dataset.py)
    return RIPanelDataset._norm_frame(shim, arr, j)


def build_encoder(args):
    """-> (embed_fn, D, backend_tag); embed_fn: (B,1,224,224) -> (B,D)."""
    if args.backend == "lingbot":
        from wpac_warning.lingbot_encoder import LingBotEncoder
        enc = LingBotEncoder(args.dir, out_dim=256, freeze=True,
                             variant=args.variant)
        D = enc.proj.in_features
        tag = f"lingbot_{args.variant}"
        embed_fn = enc.backbone_embed  # pooled pre-proj, no_grad + eval
    elif args.backend == "vit_mae":
        if not args.checkpoint:
            raise SystemExit(
                "--checkpoint is required for the vit_mae embedding cache: "
                "randomly initialised weights must never be embedded")
        from wpac_warning.mae import ViTMAEEncoder
        enc = ViTMAEEncoder(out_dim=256, checkpoint=args.checkpoint,
                            freeze=True, pool=args.pool)
        D = 384
        tag = f"vit_mae_{args.pool}"

        @torch.no_grad()
        def embed_fn(x):
            t = enc.forward_tokens(x)
            return t[:, 0] if enc.pool == "cls" else t[:, 1:].mean(dim=1)
    else:
        raise ValueError(f"unknown backend {args.backend!r}")
    return enc, embed_fn, D, tag


def list_sids(frames_dir: Path) -> list:
    return sorted(p.stem for p in frames_dir.glob("*.npy")
                  if not p.stem.endswith(("_iso", "_times")))


@torch.no_grad()
def encode_store(embed_fn, shim, arr, device, batch: int) -> np.ndarray:
    """(T_i,224,224) float16 Kelvin -> (T_i, D) float32 pooled embeds."""
    T = arr.shape[0]
    out = None
    for lo in range(0, T, batch):
        idx = range(lo, min(lo + batch, T))
        x = torch.from_numpy(np.stack(
            [norm_frame(shim, arr, j) for j in idx], axis=0)).to(device)
        z = embed_fn(x).float().cpu().numpy()
        if out is None:
            out = np.empty((T, z.shape[1]), dtype=np.float32)
        out[lo:lo + z.shape[0]] = z
    return out


def run_precompute(args, embed_fn, shim, sids, out_dir: Path) -> list:
    frames_dir = Path(args.frames_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    written = []
    for k, sid in enumerate(sids):
        out_f = out_dir / f"{sid}.npy"
        if out_f.exists() and not args.overwrite:
            continue
        arr = np.load(frames_dir / f"{sid}.npy", mmap_mode="r")
        t0 = time.time()
        emb = encode_store(embed_fn, shim, arr, device, args.batch)
        tmp = out_dir / f".{sid}.tmp.npy"
        np.save(tmp, emb.astype(np.float16))
        tmp.replace(out_f)  # atomic-ish: no partial files on crash
        written.append(sid)
        if (k + 1) % 10 == 0 or k == len(sids) - 1:
            dt = time.time() - t0
            print(f"[{k + 1}/{len(sids)}] {sid}: {emb.shape} in {dt:.1f}s",
                  flush=True)
    return written


def verify(embed_fn, shim, sids, out_dir: Path, frames_dir: Path, device,
           n: int = 5, seed: int = 0) -> float:
    """Recompute n random frames with a fresh single-frame forward and
    compare against the cached float16 rows.  Returns max abs diff."""
    rng = np.random.default_rng(seed)
    cand = []
    for sid in sids:
        emb = np.load(out_dir / f"{sid}.npy", mmap_mode="r")
        for _ in range(min(3, emb.shape[0])):
            cand.append((sid, int(rng.integers(0, emb.shape[0]))))
    rng.shuffle(cand)
    cand = cand[:n]
    arrs = {}
    max_diff = 0.0
    for sid, j in cand:
        if sid not in arrs:
            arrs[sid] = np.load(frames_dir / f"{sid}.npy", mmap_mode="r")
        x = torch.from_numpy(norm_frame(shim, arrs[sid], j)[None]).to(device)
        z = embed_fn(x).float().cpu().numpy()[0]
        zc = np.asarray(np.load(out_dir / f"{sid}.npy", mmap_mode="r")[j],
                        dtype=np.float32)
        d = float(np.abs(z - zc).max())
        max_diff = max(max_diff, d)
        print(f"  verify {sid}[{j}]: max abs diff {d:.2e}")
    return max_diff


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="optional run config; holdout enables fit-only checks")
    ap.add_argument("--backend", default="lingbot",
                    choices=["lingbot", "vit_mae"])
    ap.add_argument("--dir", default=None, help="lingbot checkpoint dir")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="vit_mae encoder checkpoint")
    ap.add_argument("--pool", default=None, choices=["cls", "mean"],
                    help="vit_mae token pooling")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--frames-index", default=None)
    ap.add_argument("--tbb-norm", default=None)
    ap.add_argument("--out", default=None,
                    help="default: data/frame_embeds/{backend_tag}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sids", default=None, help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--adopt-existing", action="store_true",
                    help="non-holdout only: adopt an existing complete cache")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)
    if args.overwrite and args.adopt_existing:
        raise SystemExit("--overwrite and --adopt-existing are mutually exclusive")

    cfg = load_cfg(args.config) if args.config else None
    holdout = bool(cfg and cfg.get("data.protocol") == "holdout_masked_v1")
    mae_audit = None
    if holdout and args.adopt_existing:
        raise SystemExit(
            "--adopt-existing is forbidden for holdout; fully re-encode the cache")
    if cfg:
        args.dir = args.dir or cfg.get("model.vision.lingbot_dir")
        args.variant = args.variant or cfg.get("model.vision.lingbot_variant")
        args.pool = args.pool or cfg.get("model.vision.mae_pool")
        if args.backend == "vit_mae" and not args.checkpoint:
            args.checkpoint = cfg.get("model.vision.mae_checkpoint")
    args.dir = resolved(args.dir or "external/lingbot")
    args.variant = args.variant or "small"
    args.pool = args.pool or "cls"
    if args.checkpoint:
        args.checkpoint = resolved(args.checkpoint)
    frames_dir = resolved(
        args.frames_dir or (cfg.get("data.frames_dir") if cfg else None)
        or P2_ROOT / "data" / "frames")
    frames_index = resolved(
        args.frames_index or (cfg.get("data.frames_index") if cfg else None)
        or P2_ROOT / "data" / "frames_index.parquet")
    norm_path = resolved(
        args.tbb_norm or (cfg.get("data.tbb_norm") if cfg else None)
        or P2_ROOT / "data" / "tbb_norm.json")
    args.frames_dir = str(frames_dir)
    args.tbb_norm = str(norm_path)
    if cfg:
        fit_only_norm_sha256(cfg, norm_path)
    # every backend encodes the QC corpus only, and the CLI frame dir must be
    # exactly the configured one (no silent corpus switch)
    qc_manifest = (cfg.get("data.qc_manifest") if cfg else None)
    if qc_manifest:
        configured_frames = resolved(cfg.get("data.frames_dir"))
        if frames_dir.resolve() != configured_frames:
            raise SystemExit(
                f"--frames-dir {frames_dir} is not data.frames_dir "
                f"({configured_frames}); the embedding cache is bound to the "
                f"configured QC corpus")
        qc_path = resolved(qc_manifest)
        if not qc_path.is_file():
            raise SystemExit(f"QC manifest missing: {qc_path}")
        qc_block = json.loads(qc_path.read_text(encoding="utf-8"))
        try:
            validate_qc_manifest(qc_block, expected_out_frames=frames_dir)
        except ValueError as exc:
            raise SystemExit(f"QC corpus validation failed: {exc}") from exc
    if holdout and args.backend == "vit_mae":
        if not args.checkpoint:
            raise SystemExit(
                "--checkpoint is required for the vit_mae embedding cache: "
                "randomly initialised weights must never be embedded")
        vision_cfg = cfg.to_dict().setdefault("model", {}).setdefault("vision", {})
        vision_cfg["backend"] = "vit_mae"
        vision_cfg["mae_checkpoint"] = str(args.checkpoint)
        mae_audit = verify_mae_for_config(cfg)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}; backend: {args.backend}")
    enc, embed_fn, D, tag = build_encoder(args)
    enc.to(device)
    shim = load_norm_shim(args.tbb_norm)
    out_dir = resolved(args.out) if args.out else (
        P2_ROOT / "data" / "frame_embeds" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = (resolved(args.checkpoint) if args.backend == "vit_mae"
                    else resolved(args.dir) / "model.pt")
    provenance = make_provenance(
        args.backend, tag,
        args.variant if args.backend == "lingbot" else None,
        args.pool if args.backend == "vit_mae" else None,
        D, frames_dir, frames_index, weights_path, norm_path,
        qc_manifest=(resolved(qc_manifest) if qc_manifest else None),
        mae_corpus=(mae_audit or {}).get("mae_corpus"))
    contract = {"provenance": provenance, "frames_dir": frames_dir,
                "embed_dir": out_dir, "holdout": holdout}

    sids = list_sids(frames_dir)
    if args.sids:
        keep = set(args.sids.split(","))
        sids = [s for s in sids if s in keep]
    if args.limit:
        sids = sids[:args.limit]
    if not sids:
        raise SystemExit("no frame stores selected")
    present = sorted(p.stem for p in out_dir.glob("*.npy")
                     if not p.name.startswith("."))
    print(f"{len(sids)} storm stores; D={D}; out: {out_dir}")

    if args.adopt_existing:
        missing = sorted(set(sids) - set(present))
        if missing:
            raise SystemExit(
                f"cannot adopt an incomplete cache; missing {len(missing)} stores "
                f"(first 10: {missing[:10]})")
    elif args.overwrite:
        extra = sorted(set(present) - set(sids))
        if extra:
            raise SystemExit(
                "subset --overwrite would leave old unverified stores; use a "
                "full overwrite or a new --out directory")
    elif present:
        validate_contract(contract, required_sids=[], allow_partial=True)
        print(f"embedding cache provenance and per-SID fingerprints verified "
              f"({out_dir / 'manifest.json'})")

    if not args.verify_only:
        if args.adopt_existing:
            written = []
            print("skip encoding: adopting the existing non-holdout cache")
        else:
            t0 = time.time()
            written = run_precompute(args, embed_fn, shim, sids, out_dir)
            dt = time.time() - t0
            print(f"wrote {len(written)} stores in {dt:.0f}s "
                  f"(skipped {len(sids) - len(written)} existing)")
        write_manifest(provenance, frames_dir, out_dir,
                       adopted_existing=args.adopt_existing)
        print(f"manifest -> {out_dir / 'manifest.json'}")
    validate_contract(contract, required_sids=sids, allow_partial=False)

    if not args.skip_verify:
        have = [s for s in sids if (out_dir / f"{s}.npy").exists()]
        md = verify(embed_fn, shim, have, out_dir, frames_dir, device)
        ok = np.allclose(0.0, md, rtol=1e-2, atol=1e-2)
        print(f"verify: max abs diff {md:.2e} -> "
              f"{'OK' if ok else 'FAIL'} (fp16 storage rounding)")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
