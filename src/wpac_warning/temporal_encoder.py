"""Temporal aggregation wrapper for the vision branch (Plan A: per-frame
encode + temporal aggregate).

Wraps any existing per-frame vision encoder (LingBotEncoder, ViTMAEEncoder).
Each frame of the (B, T, 1, 224, 224) stack is encoded independently by the
base encoder (its frozen/no_grad semantics are preserved — the base forward
is called unchanged), giving (B, T, d).  A learnable temporal position
embedding (frame order is fixed: lags ascending, t0 LAST) is added, a small
transformer encoder mixes the frames (ir_mask as key_padding_mask so
t0-filled placeholder frames never enter attention), and the t0 token (last
position) is returned as z_ir — same dimensionality as before, so the
fusion / A7 counterfactual z_ir-sharing paths are untouched.

n_frames == 1 (or a 4-D input) degenerates to an exact passthrough of the
base encoder.

embeds_input mode (offline frame-embedding cache, data.ir_embeds_dir): the
forward also accepts (B, T, D) precomputed FROZEN backbone embeddings
(pre-proj pooled vectors); they go through the SAME base.proj instance and
aggregator as the image path, so one model object can be fed either input
kind (dispatch is on input rank: 3-D = embeds, 4/5-D = images) — used for
equivalence checks and for training ~10x faster with a frozen backbone.

Optional literature-motivated variants (docs/MULTIFRAME_RS_METHODS.md;
all default OFF, old behaviour bit-identical):
  - temporal_diff (feature-differencing, FC-Siam-Diff/ChangeFormer line):
    the aggregator input becomes Linear(2d->d)(concat(z_t, z_t - z_{t-1}))
    with a zero diff vector for frame 0, computed BEFORE the position
    embedding; a frame's diff is ZEROED unless both it and its predecessor
    are real frames (ir_mask), so a t0-filled slot never contaminates a
    real frame's diff (masked frames themselves still never enter
    attention/pooling).
  - temporal_dim > 0 (slim aggregator, BreizhCrops line — at our sample
    size the temporal head should be SMALL): per-frame features are
    projected Linear(d->temporal_dim) before aggregation and the readout
    is projected back Linear(temporal_dim->d) for the fusion.
  - temporal_pool="attnpool" (attention-pooling readout instead of the
    transformer + t0-token readout): no t_encoder is built; a single-head
    attention pool with a LEARNABLE query vector scores the (position-
    embedded) frame features, softmax with the same key_padding_mask
    (masked frames get exactly zero weight), weighted sum as readout.
    The last weights are exposed as `last_pool_weights` (detached).
Pipeline order: per-frame z (d) -> [diff fuse] -> [down to temporal_dim]
-> + t_pos_embed -> pool (token: transformer + t0 token | attnpool:
learnable-query weighted sum) -> [up to d] -> z_ir.

Unknown attributes (backbone, proj, ft_mode, set_frozen_override,
depth_param_groups, ...) are delegated to the base encoder, so the train.py
machinery (backbone_lr_scale, LLRD, LP-FT) keeps working through the wrapper.
The aggregator's own parameters intentionally fall into the main-lr group,
like the other fresh heads.  They live on t_-prefixed attributes
(t_pos_embed, t_encoder) so they can never collide with a base-encoder
attribute through this delegation (ViTMAEEncoder has its own pos_embed —
a bare "pos_embed" here made nn.Module.register_parameter raise
"attribute already exists" at construction).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TEMPORAL_POOLS = ("token", "attnpool")


class TemporalAggregatorEncoder(nn.Module):
    def __init__(self, base: nn.Module, n_frames: int, d_model: int = 256,
                 n_layers: int = 1, nhead: int = 4, dropout: float = 0.1,
                 embeds_input: bool = False, temporal_diff: bool = False,
                 temporal_dim: int = 0, temporal_pool: str = "token"):
        super().__init__()
        if temporal_pool not in TEMPORAL_POOLS:
            raise ValueError(
                f"unknown temporal_pool {temporal_pool!r} "
                f"(expected one of {TEMPORAL_POOLS})")
        if int(temporal_dim) < 0:
            raise ValueError(
                f"temporal_dim must be >= 0 (0 = aggregate at the full "
                f"dimension), got {temporal_dim}")
        self.base = base
        self.n_frames = int(n_frames)
        self.out_dim = getattr(base, "out_dim", d_model)
        # config intent only; runtime dispatch is on input rank so the same
        # instance handles both image and embedding inputs
        self.embeds_input = bool(embeds_input)
        self.temporal_diff = bool(temporal_diff)
        self.agg_dim = int(temporal_dim) if int(temporal_dim) > 0 else self.out_dim
        self.temporal_pool = temporal_pool
        if self.n_frames > 1:
            if self.temporal_diff:
                # concat(z_t, z_t - z_{t-1}) -> d (frame 0 diff = zeros)
                self.t_diff_fuse = nn.Linear(2 * self.out_dim, self.out_dim)
            if self.agg_dim != self.out_dim:
                self.t_down = nn.Linear(self.out_dim, self.agg_dim)
                self.t_up = nn.Linear(self.agg_dim, self.out_dim)
            self.t_pos_embed = nn.Parameter(
                torch.randn(1, self.n_frames, self.agg_dim) * 0.02)
            if self.temporal_pool == "token":
                layer = nn.TransformerEncoderLayer(
                    self.agg_dim, nhead, dim_feedforward=2 * self.agg_dim,
                    dropout=dropout, batch_first=True)
                self.t_encoder = nn.TransformerEncoder(layer, n_layers)
            else:  # attnpool: single-head learnable-query pooling, no encoder
                self.t_pool_query = nn.Parameter(
                    torch.randn(self.agg_dim) * 0.02)

    def __getattr__(self, name: str):
        # delegate backbone/proj/ft_mode/set_frozen_override/
        # depth_param_groups etc. to the wrapped base encoder
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base")
            if base is None:
                raise
            return getattr(base, name)

    def forward(self, x: torch.Tensor,
                ir_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            # (B, T, D) cached frozen backbone embeddings (pre-proj): the
            # SAME base.proj instance as the image path is applied per frame
            B, T = x.shape[0], x.shape[1]
            z = self.base.proj(x.reshape(B * T, -1)).view(B, T, -1)
        elif x.dim() == 4 or self.n_frames == 1:
            # (B, 1, 224, 224) legacy single-frame: exact passthrough
            return self.base(x)
        else:
            # (B, T, 1, 224, 224) image stack: per-frame backbone + proj
            B, T = x.shape[0], x.shape[1]
            z = self.base(x.reshape(B * T, *x.shape[2:])).view(B, T, -1)
        if T > 1 and T != self.n_frames:
            raise ValueError(
                f"temporal stack has T={T} frames but this aggregator was "
                f"built for n_frames={self.n_frames} — check that "
                f"data.ir_lag_steps and model.vision.temporal_lags agree")

        # (1) optional feature differencing (z_t, z_t - z_{t-1}) -> d
        if self.temporal_diff:
            dz = torch.zeros_like(z)
            dz[:, 1:] = z[:, 1:] - z[:, :-1]
            if ir_mask is not None:
                # a diff is real signal only when BOTH frames are real;
                # against a t0-filled slot it would be a z_t0-relative
                # difference disguised as a temporal one
                keep = (ir_mask[:, 1:] * ir_mask[:, :-1]).unsqueeze(-1)
                dz[:, 1:] = dz[:, 1:] * keep
            z = self.t_diff_fuse(torch.cat([z, dz], dim=-1))
        # (2) optional slim projection into the aggregation dimension
        if self.agg_dim != self.out_dim:
            z = self.t_down(z)
        z = z + self.t_pos_embed[:, :T]

        kpm = None
        if ir_mask is not None:
            # key_padding_mask: True = padding (ignored by attention).
            # The t0 token (last) is never masked — it is the readout and
            # is always a real frame slot (mask semantics: 1=real, 0=fill).
            kpm = ir_mask <= 0
            kpm = kpm.clone()
            kpm[:, -1] = False

        # (3) temporal readout
        if self.temporal_pool == "token":
            z = self.t_encoder(z, src_key_padding_mask=kpm)
            out = z[:, -1]                                  # t0 token (B, d)
        else:
            scores = torch.einsum("btd,d->bt", z, self.t_pool_query)
            scores = scores / math.sqrt(self.agg_dim)
            if kpm is not None:
                scores = scores.masked_fill(kpm, float("-inf"))
            w = torch.softmax(scores, dim=-1)               # (B, T), sums to 1
            self.last_pool_weights = w.detach()
            out = torch.einsum("bt,btd->bd", w, z)
        # (4) back to the fusion dimension
        if self.agg_dim != self.out_dim:
            out = self.t_up(out)
        return out
