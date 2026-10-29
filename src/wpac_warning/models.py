# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Phase-2 model: free env encoder + Causal-GNN prior channel + vision
encoder + gated cross-attention fusion + dual heads (DV24 regression, RI
classification), with the causal loss terms from the R2 design (v2 revision).

Branches
--------
- EnvEncoder: free MLP over the full env history (B, H, N) — the main,
  unconstrained environmental DL channel; it learns whatever is predictive.
- CausalGNN: PyG GATv2 over the PCMCI+ prior edge set (message passing is
  restricted to prior edges = the causal mask; add_self_loops=False, the
  residual connection carries the self path).  Node = one environmental
  variable (feature = its H-step history) PLUS one learnable target readout
  node (B1): it has no input features (learnable embedding), no outgoing
  edges, and receives messages along the prior env -> DV24_YJ edges (the
  masked baseline has four from two variables; never hard-code the count).  The
  graph readout is [target-node state, env-node mean] concatenated.
- VisionEncoder: selectable via config model.vision.backend —
    * "lingbot": LingBot-Vision distilled student loaded from external
      weights (small/base/large = ViT-S/B/L-16, see src/lingbot_encoder.py);
      the production ensemble uses the base variant;
    * "vit_mae": ViT-S/16 MAE encoder pretrained in-domain on the HURSAT
      frame corpus (see scripts/pretrain_mae.py), [CLS] token
      projected to d_model; full finetuning by default;
    * "mini_cnn": lightweight ConvNeXt-style CNN trained from scratch
      (kept as the ablation arm; a SatMAE+LoRA backend is not used — SatMAE
      is CC BY-NC 4.0 and its multispectral patch-embed does not match
      single-channel TBB).
  Input: (1, 224, 224) raw TBB.
- GatedCrossAttention: IR embedding = query, [env, causal] embeddings = the
  two KV tokens; a sigmoid gate g lets the model fall back to the vision
  branch when the priors mislead.

Causal losses (weighted, see config model.causal_loss):
  1. adjacency cross-entropy: for every node, the distribution of the last
     GATv2 layer's attention over its incoming prior edges (ALL lags; heads
     averaged) is matched to the prior target distribution proportional to
     |coef| (B2 — replaces the old per-edge BCE towards target=1);
  2. sign consistency between each treatment's input-gradient and its
     sensitivity-checked ATE sign;
  3. do-intervention counterfactual consistency for ALL sensitivity-checked
     ATE treatments (A7): do(x_v = -sign(ATE_v) * 1 sigma) must decrease the
     predicted DV24, i.e. softplus(+delta_v) is minimized when delta_v < 0.
  (The NOTEARS acyclicity term was removed: the lag-0 prior subgraph has only
  5 fixed edges, so h(W) is identically ~0 and carried no signal — A6.)

Optional soft edge gate (B4, config model.soft_edge_gate, default OFF):
  a learnable per-edge logit; gate = sigmoid(logit) scales the prior edge
  attribute vector entering the GATv2 attention scorer, i.e. the gate soft-
  modulates attention (it does not delete edges).  Logits are initialized so
  gate ~= normalized |coef|, reproducing the un-gated model at init.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P2_ROOT / "src"))

try:
    from torch_geometric.nn import GATv2Conv
except Exception:  # pragma: no cover
    GATv2Conv = None


# ----------------------------------------------------------------------
# Branch A: Causal GNN (with learnable DV24 target readout node)
# ----------------------------------------------------------------------

class CausalGNN(nn.Module):
    """GATv2 over the prior edge set; message passing restricted to prior
    edges (the causal mask).

    The graph has n_env environmental nodes plus ONE learnable target readout
    node (index n_env): it receives the prior env -> DV24_YJ edges and has no
    input features and no outgoing edges, so env -> env message passing is
    exactly as before.  Readout = concat(target node state, env node mean).
    """

    def __init__(self, n_env: int, hist: int, hidden: int = 128,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1,
                 edge_dim: int = 3, soft_edge_gate: bool = False,
                 edge_gate_prior: Optional[torch.Tensor] = None):
        super().__init__()
        if GATv2Conv is None:
            raise RuntimeError("torch_geometric not installed")
        self.n_env = n_env
        self.n_nodes = n_env + 1  # +1 target readout node
        self.hist = hist
        self.node_embed = nn.Linear(hist, hidden)
        # learnable initial state of the target readout node (no input feats)
        self.target_embed = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                        edge_dim=edge_dim, dropout=dropout,
                                        add_self_loops=False))
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = 2 * hidden  # [target readout, env mean]

        # B4 optional soft edge gate (default off).  One logit per prior edge;
        # gate = sigmoid(logit) multiplies the edge attribute vector that
        # enters the GATv2 attention scorer.
        self.soft_edge_gate = bool(soft_edge_gate)
        if self.soft_edge_gate:
            if edge_gate_prior is None or edge_gate_prior.numel() == 0:
                raise ValueError("soft_edge_gate requires edge_gate_prior (|coef| per edge)")
            g0 = edge_gate_prior.float().clone()
            g0 = g0 / g0.max().clamp_min(1e-9)
            g0 = g0.clamp(1e-3, 1.0 - 1e-3)
            self.edge_logits = nn.Parameter(torch.log(g0 / (1 - g0)))

    def _batch_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                     batch_size: int, device) -> tuple:
        """Tile the fixed prior graph across a batch of disconnected graphs."""
        n = self.n_nodes
        ei_list, ea_list = [], []
        for b in range(batch_size):
            ei_list.append(edge_index + b * n)
            ea_list.append(edge_attr)
        ei = torch.cat(ei_list, dim=1).to(device)
        ea = torch.cat(ea_list, dim=0).to(device) if edge_attr.numel() else None
        return ei, ea

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """x: (B, H, n_env) -> z_c: (B, 2*hidden), plus last-layer attention
        over ALL prior edges for the adjacency-CE loss."""
        B = x.shape[0]
        device = x.device
        # node features: (B, n_env, hist) -> append the learnable target node
        xn = self.node_embed(x.permute(0, 2, 1))          # (B, n_env, hidden)
        tgt = self.target_embed.unsqueeze(0).expand(B, -1, -1)  # (B, 1, hidden)
        h = torch.cat([xn, tgt], dim=1).reshape(B * self.n_nodes, -1)

        if self.soft_edge_gate:
            gate = torch.sigmoid(self.edge_logits.to(device))   # (E,)
            edge_attr = edge_attr * gate.unsqueeze(-1)
        ei, ea = self._batch_edges(edge_index, edge_attr, B, device)

        attn_edges = None
        for i, conv in enumerate(self.convs):
            last = i == len(self.convs) - 1
            if last:
                h_new, (_, alpha) = conv(h, ei, edge_attr=ea,
                                         return_attention_weights=True)
                # alpha: (B*E, heads); no self-loops, so its rows follow the
                # tiled prior-edge order exactly and edge positions are
                # preserved (robust to the same (src, dst) pair appearing at
                # several lags — 21 such duplicate pairs exist).
                attn_edges = alpha
            else:
                h_new = conv(h, ei, edge_attr=ea)
            h = F.elu(h_new) + h if h.shape == h_new.shape else F.elu(h_new)
        h = self.norm(h).view(B, self.n_nodes, -1)
        h_tgt = h[:, self.n_env, :]          # (B, hidden) target readout node
        h_env = h[:, :self.n_env, :].mean(dim=1)  # (B, hidden) env mean
        hg = torch.cat([h_tgt, h_env], dim=-1)    # (B, 2*hidden)
        return hg, attn_edges


# ----------------------------------------------------------------------
# Branch A2: free environmental encoder (unconstrained DL channel)
# ----------------------------------------------------------------------

class EnvEncoder(nn.Module):
    """Free MLP over the full env history (B, H, N) — the main environmental
    DL channel.  Unlike the causal GNN it is NOT restricted to prior edges;
    the causal channel is added separately via the fusion."""

    def __init__(self, n_nodes: int, hist: int, hidden: int = 256,
                 out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_nodes * hist, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Branch B: Vision encoder (lightweight CNN baseline; SatMAE-LoRA later)
# ----------------------------------------------------------------------

class ConvNeXtTiny(nn.Module):
    """Small ConvNeXt-style CNN for (1, 224, 224) TBB frames."""

    def __init__(self, in_ch: int = 1, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=8, stride=8),  # 224 -> 28
            nn.GroupNorm(4, 48),
        )

        def block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, groups=c),
                nn.GroupNorm(4, c),
                nn.Conv2d(c, c * 2, 1),
                nn.GELU(),
                nn.Conv2d(c * 2, c, 1),
            )

        self.stages = nn.Sequential(
            block(48), block(48),
            nn.Conv2d(48, 96, 2, stride=2),  # 28 -> 14
            block(96), block(96),
            nn.Conv2d(96, 192, 2, stride=2),  # 14 -> 7
            block(192),
        )
        self.head_norm = nn.LayerNorm(192)
        self.proj = nn.Linear(192, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stages(h)
        h = h.mean(dim=(2, 3))  # global avg pool
        h = self.head_norm(h)
        return self.proj(h)


# ----------------------------------------------------------------------
# Gated cross-attention fusion
# ----------------------------------------------------------------------

class GatedCrossAttention(nn.Module):
    """IR embedding as query, [env, causal] embeddings as the KV tokens;
    sigmoid gate falls back to the vision branch when the environmental /
    causal priors are uninformative or misleading.  n_ctx=1 (causal branch
    disabled) reduces the KV set to the env token alone."""

    def __init__(self, d_model: int = 256, heads: int = 4, n_ctx: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.n_ctx = n_ctx
        self.gate = nn.Sequential(nn.Linear(d_model * (1 + n_ctx), d_model),
                                  nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_ir: torch.Tensor, z_env: torch.Tensor,
                z_causal: Optional[torch.Tensor] = None):
        ctx = [z_env] + ([z_causal] if z_causal is not None else [])
        q = z_ir.unsqueeze(1)                    # (B,1,d) query = vision
        kv = torch.stack(ctx, dim=1)             # (B,n_ctx,d)
        att, aw = self.attn(q, kv, kv)
        att = att.squeeze(1)
        g = self.gate(torch.cat([z_ir] + ctx, dim=-1))
        out = self.norm(z_ir + g * att)
        causal_share = (aw.squeeze(1)[:, -1] if z_causal is not None
                        else torch.zeros_like(aw.squeeze(1)[:, -1]))
        return out, g, causal_share


# ----------------------------------------------------------------------
# Full model with dual heads
# ----------------------------------------------------------------------

class CausalRINet(nn.Module):
    def __init__(self, cfg, prior: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg
        n_nodes = len(cfg.get("data.env_vars"))
        hist = int(cfg.get("data.history_steps", 5))
        dm = int(cfg.get("model.fusion.d_model", 256))
        g = cfg.get("model.gnn", {})
        v = cfg.get("model.vision", {})
        e = cfg.get("model.env_encoder", {})
        soft_gate = bool(cfg.get("model.soft_edge_gate", False))
        gate_prior = prior["edge_attr"][:, 2] if (soft_gate and prior is not None) else None
        self.gnn_enabled = bool(g.get("enabled", True))
        if self.gnn_enabled:
            self.gnn = CausalGNN(n_nodes, hist,
                                 hidden=int(g.get("hidden", 128)),
                                 heads=int(g.get("heads", 4)),
                                 layers=int(g.get("layers", 2)),
                                 dropout=float(g.get("dropout", 0.1)),
                                 soft_edge_gate=soft_gate,
                                 edge_gate_prior=gate_prior)
        self.env_enc = EnvEncoder(n_nodes, hist,
                                  hidden=int(e.get("hidden", 256)), out_dim=dm)
        # vision branch: "lingbot" (external LingBot-Vision student; the
        # production ensemble uses base = ViT-B/16), "vit_mae" (config
        # default; domain-pretrained ViT-S/16 MAE encoder) or "mini_cnn"
        # (lightweight CNN from scratch, ablation arm)
        backend = str(v.get("backend", "vit_mae"))
        self.vision_backend = backend
        if backend == "vit_mae":
            if (str(cfg.get("data.protocol", "")) == "holdout_masked_v1"
                    and not v.get("mae_checkpoint")):
                raise ValueError(
                    "holdout protocol requires model.vision.mae_checkpoint; "
                    "randomly initialised MAE weights are not allowed")
            from wpac_warning.mae import ViTMAEEncoder
            self.vision = ViTMAEEncoder(
                out_dim=dm,
                checkpoint=v.get("mae_checkpoint"),
                freeze=bool(v.get("freeze_backbone", False)),
                pool=str(v.get("mae_pool", "cls")),
                img_size=int(v.get("img_size", 224)),
                in_chans=int(v.get("in_channels", 1)))
        elif backend == "lingbot":
            from wpac_warning.lingbot_encoder import LingBotEncoder
            self.vision = LingBotEncoder(
                ckpt_dir=str(v.get("lingbot_dir", "external/lingbot")),
                out_dim=dm,
                freeze=bool(v.get("freeze_backbone", True)),
                variant=str(v.get("lingbot_variant", "small")),
                ft_mode=v.get("ft_mode"),
                ft_blocks=int(v.get("ft_blocks", 4)),
                lora_r=int(v.get("lora_r", 16)),
                lora_alpha=float(v.get("lora_alpha", 32.0)))
        elif backend in ("mini_cnn", "convnext_tiny"):  # legacy name alias
            self.vision = ConvNeXtTiny(in_ch=int(v.get("in_channels", 1)), out_dim=dm)
        else:
            raise ValueError(f"unknown vision backend: {backend}")
        # multi-frame temporal aggregation (Plan A): when
        # model.vision.temporal_lags is non-empty the per-frame encoder is
        # wrapped in TemporalAggregatorEncoder (per-frame encode -> temporal
        # transformer over the frame stack, t0 token as z_ir).  Default empty
        # = legacy single-frame behaviour (no wrapper).
        tlags = v.get("temporal_lags") or []
        # the frame stack is configured twice (dataset side and model side)
        # and the two MUST be the same sequence: a mismatch either puts the
        # t0 readout on the wrong position embedding (different lengths) or
        # silently bypasses aggregation (dataset single-frame vs wrapper)
        ir_lags = [int(l) for l in (cfg.get("data.ir_lag_steps", []) or [])]
        tlags = [int(l) for l in tlags]
        if ir_lags != tlags:
            raise ValueError(
                f"data.ir_lag_steps {ir_lags} != model.vision.temporal_lags "
                f"{tlags}: configure the dataset frame stack and the temporal "
                f"wrapper with the same lag sequence (both empty = single-frame)")
        self._vision_temporal = bool(tlags) and backend in ("lingbot", "vit_mae")
        if self._vision_temporal:
            from wpac_warning.temporal_encoder import TemporalAggregatorEncoder
            self.vision = TemporalAggregatorEncoder(
                self.vision, n_frames=1 + len(tlags), d_model=dm,
                n_layers=int(v.get("temporal_layers", 1)),
                nhead=int(v.get("temporal_heads", 4)),
                dropout=float(v.get("temporal_dropout", 0.1)),
                # offline frame-embedding cache: forward also accepts
                # (B, T, D) pre-proj backbone embeds (same proj + aggregator)
                embeds_input=bool(str(cfg.get("data.ir_embeds_dir", "") or "")),
                # literature-motivated variants (docs/MULTIFRAME_RS_METHODS.md;
                # all default off = legacy behaviour)
                temporal_diff=bool(v.get("temporal_diff", False)),
                temporal_dim=int(v.get("temporal_dim", 0)),
                temporal_pool=str(v.get("temporal_pool", "token")))
        # project gnn to d_model if needed
        if self.gnn_enabled:
            self.gnn_proj = nn.Linear(self.gnn.out_dim, dm) if self.gnn.out_dim != dm else nn.Identity()
        self.fusion = GatedCrossAttention(d_model=dm,
                                          heads=int(cfg.get("model.fusion.heads", 4)),
                                          n_ctx=2 if self.gnn_enabled else 1)
        self.head_dv24 = nn.Linear(dm, 1)
        self.head_ri = nn.Linear(dm, 1)

    def forward(self, x_env: torch.Tensor, x_ir: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor,
                z_ir: Optional[torch.Tensor] = None,
                ir_mask: Optional[torch.Tensor] = None,
                env_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """z_ir: optional precomputed vision embedding (batch may differ from
        x_env's — used by the A7 counterfactual block, which shares one
        vision forward across all intervention variants).
        ir_mask: (B, T) frame mask (1=real, 0=t0-fill) for the temporal
        vision wrapper; ignored by single-frame encoders.
        env_mask: (B, H) environmental history validity, ones when omitted."""
        if env_mask is None:
            env_mask = torch.ones(x_env.shape[:-1], dtype=x_env.dtype,
                                  device=x_env.device)
        if tuple(env_mask.shape) != tuple(x_env.shape[:-1]):
            raise ValueError(
                f"env_mask shape {tuple(env_mask.shape)} != x_env prefix "
                f"{tuple(x_env.shape[:-1])}")
        env_mask = env_mask.to(device=x_env.device, dtype=x_env.dtype)
        x_env = x_env * env_mask.unsqueeze(-1)
        if self.gnn_enabled:
            z_c, attn_edges = self.gnn(x_env, edge_index, edge_attr)
            z_c = self.gnn_proj(z_c)
        else:
            z_c, attn_edges = None, None
        z_e = self.env_enc(x_env)
        if z_ir is not None:
            z_s = z_ir
        elif self._vision_temporal:
            z_s = self.vision(x_ir, ir_mask)
        else:
            z_s = self.vision(x_ir)
        z, gate, causal_share = self.fusion(z_s, z_e, z_c)
        return {
            "dv24": self.head_dv24(z).squeeze(-1),
            "ri_logit": self.head_ri(z).squeeze(-1),
            "gate": gate,
            "causal_attn_share": causal_share,
            "z_gnn": z_c,
            "z_env": z_e,
            "z_ir": z_s,
            "attn_edges": attn_edges,  # (B*E, heads) last-layer GATv2 attention
        }

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Causal losses
# ----------------------------------------------------------------------

class CausalLossTerms(nn.Module):
    """Causal regularizers, computed against the prior graph and the
    sensitivity-checked ATE signs (from phase-1 ate_results.csv):

    - adjacency_ce: per-node cross-entropy between last-layer GATv2 attention
      over incoming prior edges (all lags, heads averaged) and the prior
      target distribution p ~ |coef| normalized over that node's in-edges (B2);
    - sign consistency of input gradients with ATE signs;
    - counterfactual consistency for every trusted ATE treatment (A7).
    """

    def __init__(self, cfg, prior: dict, ate_signs: Dict[str, float]):
        super().__init__()
        self.cfg = cfg
        self.prior = prior
        self.ate_signs = ate_signs  # {treatment: +1/-1} from sensitivity-checked ATEs
        w = cfg.get("model.causal_loss", {})
        self.w_adj = float(w.get("w_adj_ce", 0.05))
        self.w_sign = float(w.get("w_sign_ate", 0.05))
        self.w_cf = float(w.get("w_counterfactual", 0.05))

        # B2 target distributions: for every node with incoming prior edges,
        # p_e = |coef_e| / sum_{in-edges} |coef| (positions index the single-
        # graph edge list; alpha rows follow the same order, batch-tiled).
        ei = prior["edge_index"]
        ea = prior["edge_attr"]
        self.n_edges = ei.shape[1]
        self._dst_groups: List[tuple] = []  # (positions LongTensor, p FloatTensor)
        if self.n_edges:
            dsts = ei[1].unique()
            for d in dsts:
                pos = (ei[1] == d).nonzero(as_tuple=True)[0]
                wc = ea[pos, 2].float().clamp_min(0)
                p = wc / wc.sum().clamp_min(1e-9)
                self._dst_groups.append((pos, p))

    def adjacency_ce(self, attn_edges: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean over nodes of CE(p_prior || q_attn), where q is the attention
        distribution over the node's incoming prior edges (batch- and head-
        averaged, renormalized within the in-edge set)."""
        if (attn_edges is None or attn_edges.numel() == 0
                or not self._dst_groups or self.n_edges == 0):
            return torch.tensor(0.0, requires_grad=True)
        a = attn_edges.mean(dim=-1)  # (B*E,)
        if a.numel() % self.n_edges != 0:
            return torch.tensor(0.0, requires_grad=True)
        a = a.view(-1, self.n_edges)  # (B, E)
        losses = []
        for pos, p in self._dst_groups:
            pos = pos.to(a.device)
            p = p.to(a.device)
            q = a[:, pos]
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-9)
            ce = -(p * (q + 1e-9).log()).sum(dim=1)  # (B,)
            losses.append(ce.mean())
        return torch.stack(losses).mean()

    def sign_consistency(self, grads: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Penalize sign mismatch between input gradients and ATE signs."""
        losses = []
        for trt, sign in self.ate_signs.items():
            g = grads.get(trt)
            if g is None:
                continue
            losses.append(F.softplus(-float(sign) * g.mean()))
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()

    def counterfactual(self, deltas: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A7: the intervention moves every trusted treatment against its ATE
        sign, so the predicted DV24 change is expected to be NEGATIVE."""
        losses = [F.softplus(d.mean()) for d in deltas.values() if d is not None]
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()

    def forward(self, grads, attn_edges, deltas) -> Dict[str, torch.Tensor]:
        parts = {
            "adj_ce": self.adjacency_ce(attn_edges),
            "sign": self.sign_consistency(grads),
            "counterfactual": self.counterfactual(deltas),
        }
        total = (self.w_adj * parts["adj_ce"] + self.w_sign * parts["sign"]
                 + self.w_cf * parts["counterfactual"])
        parts["total"] = total
        return parts
