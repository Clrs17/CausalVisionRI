"""RITCF-recon models: the siamese contrastive net and the direct-classifier control.

Architecture (spec §4) -- all of it is a reconstruction, not a transcription:

* SAT branch    : 5 frames x 224 x 224 -> Conv3x3(32)+ReLU+MaxPool x3 -> FC 128
* ERA5 branch   : 85 channels x 25 x 25 -> Conv3x3(32)+ReLU+MaxPool x2 -> FC 128
* HIS branch    : 30 -> MLP -> 64
* concat (320)  : FC 64 -> 32-d embedding (linear output, as in the paper's Part C)
* Part C        : d = ||f(A) - f(B)||_2, trained with MSE against label 0 (RI-RI)
                  and 1 (cross-class)

The paper never states the conv widths, kernel count, FC widths, whether the two
branches share weights, or whether BatchNorm is used; the defaults here are the
spec's choices, and the switches below expose the two the variants need:

* ``shared_weights=False`` + ``scalar_distance=True`` reproduces the published
  skeleton repository (unshared encoders, scalar output, |f(a)-f(b)|).
* ``direct=True`` drops the siamese path entirely: the same encoder feeds a
  single linear logit, trained with BCE -- the V4 contrastive-learning control.

``use_bn`` defaults to False because the paper never mentions BatchNorm.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SatEncoder(nn.Module):
    """Five TBB frames as channels, three conv/pool stages, FC 128."""

    def __init__(self, frames: int = 5, size: int = 224, width: int = 32,
                 out_dim: int = 128, use_bn: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvStage(frames, width, use_bn),
            ConvStage(width, width, use_bn),
            ConvStage(width, width, use_bn),
        )
        feat = width * (size // 8) * (size // 8)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class Era5Encoder(nn.Module):
    """All (variable x level x time) channels, two conv/pool stages, FC 128."""

    def __init__(self, channels: int = 85, size: int = 25, width: int = 32,
                 out_dim: int = 128, use_bn: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvStage(channels, width, use_bn),
            ConvStage(width, width, use_bn),
        )
        feat = width * (size // 4) * (size // 4)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, out_dim), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class HisEncoder(nn.Module):
    def __init__(self, in_dim: int = 30, out_dim: int = 64, use_bn: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Encoder(nn.Module):
    """One side of the siamese pair (Input-A or Input-B) -> embedding."""

    def __init__(self, frames: int = 5, sat_size: int = 224, grid_channels: int = 85,
                 grid_size: int = 25, his_dim: int = 30, hidden: int = 64,
                 embed_dim: int = 32, use_bn: bool = False) -> None:
        super().__init__()
        self.sat = SatEncoder(frames, sat_size, 32, 128, use_bn)
        self.era5 = Era5Encoder(grid_channels, grid_size, 32, 128, use_bn)
        self.his = HisEncoder(his_dim, 64, use_bn)
        self.merge = nn.Sequential(
            nn.Linear(128 + 128 + 64, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        z = torch.cat([self.sat(batch["sat"]), self.era5(batch["era5"]),
                       self.his(batch["his"])], dim=1)
        return self.merge(z)


class RITCFSiamese(nn.Module):
    """Twin-branch module whose output is the Part-C distance."""

    def __init__(self, shared_weights: bool = True, embed_dim: int = 32,
                 scalar_distance: bool = False, use_bn: bool = False, **encoder_kw) -> None:
        super().__init__()
        self.shared_weights = shared_weights
        self.scalar_distance = scalar_distance
        self.encoder_a = Encoder(embed_dim=embed_dim, use_bn=use_bn, **encoder_kw)
        self.encoder_b = (self.encoder_a if shared_weights
                          else Encoder(embed_dim=embed_dim, use_bn=use_bn, **encoder_kw))

    def embed(self, batch: dict[str, Tensor], side: str) -> Tensor:
        return getattr(self, f"encoder_{side}")(batch)

    def forward(self, a: dict[str, Tensor], b: dict[str, Tensor]) -> Tensor:
        fa = self.encoder_a(a)
        fb = self.encoder_b(b)
        if self.scalar_distance:
            return (fa - fb).abs()                      # (B, 1)
        return torch.linalg.vector_norm(fa - fb, dim=1, keepdim=True)


class RITCFDirect(nn.Module):
    """Same encoders, no pairing: one logit per sample (V4 control)."""

    def __init__(self, embed_dim: int = 32, use_bn: bool = False, **encoder_kw) -> None:
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim, use_bn=use_bn, **encoder_kw)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.head(self.encoder(batch))


def build_model(cfg: dict) -> nn.Module:
    """Rebuild a model from a checkpoint config dict."""
    kw = dict(
        frames=cfg.get("frames", 5), sat_size=cfg.get("sat_size", 224),
        grid_channels=cfg.get("grid_channels", 85), grid_size=cfg.get("grid_size", 25),
        his_dim=cfg.get("his_dim", 30), hidden=cfg.get("hidden", 64),
        embed_dim=cfg.get("embed_dim", 32), use_bn=cfg.get("use_bn", False),
    )
    if cfg.get("direct", False):
        return RITCFDirect(**kw)
    return RITCFSiamese(shared_weights=cfg.get("shared_weights", True),
                        scalar_distance=cfg.get("scalar_distance", False), **kw)


def model_config(variant_name: str, use_bn: bool = False) -> dict:
    from scripts.ritcf_recon import config as C
    v = C.variant(variant_name)
    return {
        "variant": v.name,
        "direct": v.direct_classifier,
        "shared_weights": v.shared_weights,
        "scalar_distance": v.scalar_distance,
        "embed_dim": v.embed_dim,
        "frames": C.SAT_FRAMES, "sat_size": C.SAT_SIZE,
        "grid_channels": C.GRID_CHANNELS, "grid_size": C.GRID_SIZE,
        "his_dim": C.HIS_DIM, "hidden": 64, "use_bn": use_bn,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
