"""ViT-S/16 Masked Autoencoder (MAE), self-contained (no timm dependency).

Standard MAE design (He et al. 2022), scaled to ViT-S:
  encoder: embed_dim 384, depth 12, heads 6, patch 16, 224x224 input
           (14x14 = 196 patches + [CLS]); in_chans=1 (single-channel TBB);
  decoder: embed_dim 192, depth 4, heads 6, mask token, predicts the raw
           16x16 patch pixels of masked positions;
  mask_ratio 0.75, per-sample random masking, optional per-patch normalized
  pixel loss (norm_pix_loss).

`ViTMAEEncoder` wraps ONLY the encoder half for downstream use: it loads a
pretrain checkpoint (decoder keys ignored) and outputs the [CLS] token (or
mean-pooled patch tokens) projected to `out_dim`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, N, hd)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16,
                 in_chans: int = 1, embed_dim: int = 384):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


# ----------------------------------------------------------------------
# MAE
# ----------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16, in_chans: int = 1,
                 embed_dim: int = 384, depth: int = 12, heads: int = 6,
                 dec_dim: int = 192, dec_depth: int = 4, dec_heads: int = 6,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.75,
                 norm_pix_loss: bool = True):
        super().__init__()
        self.patch = patch
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # ---- encoder ----
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, embed_dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- decoder ----
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, n + 1, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, mlp_ratio)
                                         for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch * patch * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        for p in (self.pos_embed, self.dec_pos_embed):
            nn.init.trunc_normal_(p, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- masking ----
    def random_masking(self, x: torch.Tensor, mask_ratio: float,
                       noise: Optional[torch.Tensor] = None):
        """Per-sample shuffle masking.  x: (B, N, D).
        Returns x_visible, binary mask (1 = masked), ids_restore."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        if noise is None:
            noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # back to patch order
        return x_vis, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float,
                        noise: Optional[torch.Tensor] = None):
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)
        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.dec_embed(x)
        B, N_vis, D = x.shape
        n_mask = ids_restore.shape[1] + 1 - N_vis
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.dec_pos_embed
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_pred(self.dec_norm(x))
        return x[:, 1:, :]  # (B, N, p*p*C), no CLS

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, p*p*C)"""
        p = self.patch
        B, C, H, W = imgs.shape
        g = H // p
        x = imgs.reshape(B, C, g, p, g, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(B, g * g, p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) -> (B, C, H, W)"""
        p = self.patch
        g = int(x.shape[1] ** 0.5)
        c = x.shape[2] // (p * p)
        x = x.reshape(x.shape[0], g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(x.shape[0], c, g * p, g * p)

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) per-patch loss
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, imgs: torch.Tensor, mask_ratio: Optional[float] = None,
                noise: Optional[torch.Tensor] = None):
        mr = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mr, noise)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ---- downstream encoder view ----
    def encoder_state_dict(self) -> dict:
        """Encoder-half weights with clean keys (patch_embed/blocks/norm/...)."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(("dec_", "mask_token"))}


# ----------------------------------------------------------------------
# Downstream encoder: MAE encoder + linear projection to the fusion dim
# ----------------------------------------------------------------------

class ViTMAEEncoder(nn.Module):
    """Encoder half of MaskedAutoencoderViT for downstream RI prediction.

    Loads a MAE pretrain checkpoint (full MAE state dict; decoder keys are
    ignored).  Output: [CLS] token (default) or mean-pooled patch tokens,
    linearly projected to out_dim — same interface as the mini-CNN branch.
    """

    def __init__(self, out_dim: int = 256, checkpoint: Optional[str] = None,
                 freeze: bool = False, pool: str = "cls", img_size: int = 224,
                 in_chans: int = 1):
        super().__init__()
        mae = MaskedAutoencoderViT(img_size=img_size, in_chans=in_chans)
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed
        self.blocks = mae.blocks
        self.norm = mae.norm
        self.pool = pool
        self.proj = nn.Linear(384, out_dim)  # fresh head: always trainable
        self.out_dim = out_dim
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)  # pretrain checkpoints store {"model": ...}
            enc_keys = [k for k in self.state_dict() if not k.startswith("proj")]
            missing = [k for k in enc_keys if k not in sd]
            if missing:
                raise KeyError(f"MAE checkpoint missing encoder keys: {missing[:5]}...")
            self.load_state_dict({k: sd[k] for k in enc_keys}, strict=False)
        if freeze:
            for name, p in self.named_parameters():
                if not name.startswith("proj"):
                    p.requires_grad_(False)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Full token sequence (B, 1+N, 384) — no masking at downstream."""
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.forward_tokens(x)
        z = t[:, 0] if self.pool == "cls" else t[:, 1:].mean(dim=1)
        return self.proj(z)
