# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""LingBot-Vision (Ant Robbyant, Apache-2.0) vision backend for CausalRINet.

Loads a distilled LingBot-Vision student backbone from a LOCAL checkpoint
through the vendored lingbot_vision package (external/lingbot-vision);
the variant argument selects the size (small = ViT-S/16 embed 384, base =
ViT-B/16 embed 768, large = ViT-L/16 embed 1024; all patch 16, RoPE fp32,
4 register tokens).  The production ensemble uses the base variant.
Input: our standardized
single-channel TBB (B, 1, 224, 224), repeated to 3 channels (scale is already
zero-mean/unit-std, close to ImageNet-normalized inputs).  Patch tokens are
mean-pooled and linearly projected to out_dim, matching the fusion interface.

Frozen by default (A3 lesson: at our sample size frozen backbones beat full
finetuning); pass freeze=False for the partial/full-finetune arm.

Fine-tuning strategy matrix (lingbot backend only), selected with ft_mode:
  - None      : legacy behaviour from `freeze` (True -> frozen, False -> full)
  - "frozen"  : backbone fully frozen, eval + no_grad (== freeze=True)
  - "full"    : backbone fully trainable (== freeze=False)
  - "partial" : only the last `ft_blocks` transformer blocks + final norm are
                trainable; forward runs with gradients (the DINOv2-style
                backbone ships drop_path=attn_drop=proj_drop=0, so train()
                mode is deterministic and safe for the frozen blocks)
  - "lora"    : backbone frozen; every block.attn qkv/proj Linear is wrapped
                in LoRALinear (frozen base + trainable low-rank branch,
                B zero-init, scaling = lora_alpha / lora_r)

depth_param_groups(base_lr, decay) provides LLRD (layer-wise lr decay) groups
for the optimizer; set_frozen_override(bool) supports LP-FT (linear-probe
warmup then finetune) switching from train.py.

arXiv:2607.05247, weights: robbyant/lingbot-vision-vit-{small,base,large}
(HF/ModelScope; the production ensemble uses vit-base).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

P2_ROOT = Path(__file__).resolve().parents[2]
_REPO = P2_ROOT / "external" / "lingbot-vision"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FT_MODES = ("frozen", "full", "partial", "lora")


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank branch (Hu et al., 2021).

    out = base(x) + (alpha / r) * B(A(x));  B is zero-initialized so the
    branch starts as an exact identity on the pretrained function.
    """

    def __init__(self, base: nn.Module, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.weight.shape[1], base.weight.shape[0]
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = float(alpha) / float(r)

    def __getattr__(self, name: str):
        # the vendored SelfAttention reads qkv.in_features etc. — forward
        # unknown attributes to the wrapped base Linear
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base")
            if base is None:
                raise
            return getattr(base, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


class LingBotEncoder(nn.Module):
    def __init__(self, ckpt_dir: str, out_dim: int = 256, freeze: bool = True,
                 variant: str = "small", ft_mode: str | None = None,
                 ft_blocks: int = 4, lora_r: int = 16,
                 lora_alpha: float = 32.0):
        super().__init__()
        from lingbot_vision import load_pretrained_backbone
        backbone, embed_dim = load_pretrained_backbone(
            ckpt_dir, variant=variant, device="cpu", dtype=torch.float32,
            verbose=False)
        self.backbone = backbone
        self.proj = nn.Linear(embed_dim, out_dim)
        self.out_dim = out_dim
        # ft_mode=None keeps the legacy `freeze` semantics exactly
        if ft_mode is None:
            ft_mode = "frozen" if freeze else "full"
        if ft_mode not in FT_MODES:
            raise ValueError(f"unknown ft_mode {ft_mode!r} (expected one of {FT_MODES})")
        self.ft_mode = ft_mode
        self.ft_blocks = int(ft_blocks)
        # legacy attribute: True only when the backbone never sees gradients
        self.freeze = ft_mode == "frozen"
        self._frozen_override = False  # LP-FT warmup switch (train.py)

        if ft_mode == "lora":
            self._inject_lora(lora_r, lora_alpha)
        self._apply_trainability()

    # ------------------------------------------------------------------
    # trainability management
    # ------------------------------------------------------------------
    def _inject_lora(self, r: int, alpha: float) -> None:
        for blk in self.backbone.blocks:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha)
            blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha)

    def _apply_trainability(self) -> None:
        """Set requires_grad according to the current ft_mode."""
        mode = self.ft_mode
        for p in self.backbone.parameters():
            p.requires_grad_(mode == "full")
        if mode == "partial":
            n = len(self.backbone.blocks)
            k = min(self.ft_blocks, n)
            for blk in list(self.backbone.blocks)[n - k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)
        elif mode == "lora":
            # only the low-rank branches train (base weights were frozen at
            # wrap time; the blanket freeze above keeps them that way)
            for m in self.backbone.modules():
                if isinstance(m, LoRALinear):
                    for p in m.lora_A.parameters():
                        p.requires_grad_(True)
                    for p in m.lora_B.parameters():
                        p.requires_grad_(True)

    def set_frozen_override(self, frozen: bool) -> None:
        """LP-FT switch: True = behave exactly like ft_mode='frozen'
        (requires_grad off + eval + no_grad); False = restore this arm's
        configured ft_mode trainability.  Idempotent."""
        self._frozen_override = bool(frozen)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        else:
            self._apply_trainability()
            if self.ft_mode != "frozen":
                self.backbone.train()

    def _effective_frozen(self) -> bool:
        return self._frozen_override or self.ft_mode == "frozen"

    # ------------------------------------------------------------------
    # LLRD parameter groups (layer-wise learning-rate decay)
    # ------------------------------------------------------------------
    def depth_param_groups(self, base_lr: float, decay: float):
        """Optimizer param groups with lr = base_lr * decay**depth_factor.

        Embedding layers (patch_embed, cls/register tokens, ...) sit at
        factor L (= n_blocks, same as block 0), blocks[i] at factor L - i,
        the final norm at factor 0.  Only trainable parameters are grouped;
        `self.proj` is a head and is NOT included (main lr in train.py).
        """
        blocks = self.backbone.blocks
        L = len(blocks)
        groups = []
        embed_params = []
        for n, p in self.backbone.named_parameters():
            if not (n.startswith("blocks.") or n.startswith("norm.")):
                if p.requires_grad:
                    embed_params.append(p)
        if embed_params:
            groups.append({"params": embed_params, "lr": base_lr * decay ** L})
        for i, blk in enumerate(blocks):
            bp = [p for p in blk.parameters() if p.requires_grad]
            if bp:
                groups.append({"params": bp, "lr": base_lr * decay ** (L - i)})
        np_ = [p for p in self.backbone.norm.parameters() if p.requires_grad]
        if np_:
            groups.append({"params": np_, "lr": base_lr})
        return groups

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 224, 224) single-channel TBB -> 3ch (channel-repeat adaptation)
        x3 = x.repeat(1, 3, 1, 1)
        # NB: their API's is_training=True selects the full patch-token output
        # dict (not "training behaviour") — it must stay True in eval too.
        # Frozen mode is enforced by eval() + no_grad instead.
        if self._effective_frozen():
            self.backbone.eval()  # counteract the parent model's recursive .train()
            with torch.no_grad():
                out = self.backbone(x3, is_training=True)
        else:
            # partial/full/lora: gradients flow.  The vendored backbone is
            # built with drop_path=attn_drop=proj_drop=0 (build.py), so
            # train() mode is deterministic; frozen blocks are unaffected.
            self.backbone.train()
            out = self.backbone(x3, is_training=True)
        tok = out["x_norm_patchtokens"]          # (B, 196, 384)
        return self.proj(tok.mean(dim=1))

    @torch.no_grad()
    def backbone_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone output BEFORE self.proj, (B, embed_dim) — the
        quantity stored by the offline frame-embedding cache
        (scripts/precompute_frame_embeds.py).  Always eval + no_grad,
        regardless of ft_mode, so cached embeds stay valid for every arm."""
        self.backbone.eval()
        x3 = x.repeat(1, 3, 1, 1)
        out = self.backbone(x3, is_training=True)
        return out["x_norm_patchtokens"].mean(dim=1)
