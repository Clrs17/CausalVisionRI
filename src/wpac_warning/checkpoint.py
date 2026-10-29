"""Inference checkpoint format for HOLDOUT_MASKED_V1.

* A fold model with a FULLY FROZEN visual backbone stores only its trainable
  parameters plus a SHA-256 digest of the frozen backbone tensors.  The
  backbone is reconstructed from its external initialisation (config/weights
  manifest) and the digest is verified before inference — the checkpoint does
  not duplicate 20-90 M frozen weights.
* Any arm that trains part of the backbone (full FT, partial FT, LoRA) stores
  the complete state dict, because inference needs those adapted weights.

The format is a plain dict saved with ``torch.save``::

    {"kind": "holdout_masked_v1",
     "mode": "trainable_only" | "full",
     "trainable": {name: tensor},      # mode == trainable_only
     "frozen_keys": [str],             # keys absent from "trainable"
     "backbone": {"backend": str, "sha256": str, "n_params": int,
                  "n_frozen": int},
     "full": {name: tensor},           # mode == full
     ...meta scalars...}
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

CKPT_KIND = "holdout_masked_v1"


def _vision_base(model) -> Optional[nn.Module]:
    vision = getattr(model, "vision", None)
    if vision is None:
        return None
    # TemporalAggregatorEncoder wraps the per-frame encoder in `.base`
    return getattr(vision, "base", vision)


def backbone_module(model) -> Optional[nn.Module]:
    """The pretrained visual backbone, or None for from-scratch backends."""
    base = _vision_base(model)
    if base is None:
        return None
    backend = getattr(model, "vision_backend", "")
    if backend == "vit_mae":
        return base
    if backend == "lingbot":
        return getattr(base, "backbone", None)
    return None


def _is_head(name: str) -> bool:
    """Fresh projection heads are not part of the frozen backbone."""
    return name == "proj" or name.endswith(".proj")


def backbone_named_parameters(model) -> Dict[str, nn.Parameter]:
    bb = backbone_module(model)
    if bb is None:
        return {}
    return {n: p for n, p in bb.named_parameters() if not _is_head(n)}


def backbone_fully_frozen(model) -> bool:
    params = backbone_named_parameters(model)
    if not params:
        return False
    return not any(p.requires_grad for p in params.values())


def backbone_digest(model) -> str:
    """SHA-256 over the frozen backbone tensors (names + float32 bytes).

    Memoised on the model: the frozen backbone never changes after
    initialisation, so re-hashing it at save + manifest time is pure I/O.
    """
    cached = getattr(model, "_backbone_digest_cache", None)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    params = backbone_named_parameters(model)
    if not params:
        return ""
    for name in sorted(params):
        t = params[name].detach().to(torch.float32).cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    digest = h.hexdigest()
    try:
        model._backbone_digest_cache = digest
    except Exception:
        pass
    return digest


def _frozen_param_keys(model) -> set:
    """Fully-qualified names of parameters that are not trained."""
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def checkpoint_payload(model, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build the checkpoint dict (see module docstring)."""
    digest = backbone_digest(model)
    payload: Dict[str, Any] = {
        "kind": CKPT_KIND,
        "backbone": {
            "backend": str(getattr(model, "vision_backend", "")),
            "sha256": digest,
            "n_params": int(sum(p.numel()
                                for p in backbone_named_parameters(model).values())),
            "frozen": backbone_fully_frozen(model),
        },
    }
    if backbone_fully_frozen(model):
        frozen = _frozen_param_keys(model)
        state = model.state_dict()
        payload["mode"] = "trainable_only"
        payload["trainable"] = {k: v.detach().cpu() for k, v in state.items()
                                if k not in frozen}
        payload["frozen_keys"] = sorted(frozen)
        payload["trainable_params"] = int(sum(
            p.numel() for p in model.parameters() if p.requires_grad))
    else:
        payload["mode"] = "full"
        payload["full"] = {k: v.detach().cpu()
                           for k, v in model.state_dict().items()}
        payload["trainable_params"] = int(sum(
            p.numel() for p in model.parameters() if p.requires_grad))
    payload.update(meta)
    return payload


def save_inference_checkpoint(path: str | Path, model, meta: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, meta)
    torch.save(payload, path)
    return payload


def load_inference_checkpoint(path: str | Path, model,
                              verify_backbone: bool = True
                              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a checkpoint into `model`; returns (meta, info).

    Raises on: wrong kind, missing/mismatched trainable keys, or (when the
    checkpoint was written in trainable-only mode) a backbone digest
    mismatch.  Never falls back to a partial/optimistic load.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != CKPT_KIND:
        raise ValueError(f"{path} is not a {CKPT_KIND} checkpoint "
                         f"(kind={ckpt.get('kind')!r})")
    mode = ckpt.get("mode")
    if mode == "full":
        model.load_state_dict(ckpt["full"], strict=True)
    elif mode == "trainable_only":
        frozen = _frozen_param_keys(model)
        expected = set(model.state_dict()) - frozen
        got = set(ckpt["trainable"])
        missing = expected - got
        extra = got - expected
        if missing or extra:
            raise RuntimeError(
                f"checkpoint/model mismatch for {path.name}: "
                f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        loaded = model.load_state_dict(ckpt["trainable"], strict=False)
        allowed_missing = set(model.state_dict()) - expected
        if (set(loaded.missing_keys) != allowed_missing
                or set(loaded.unexpected_keys)):
            raise RuntimeError(
                f"checkpoint/model mismatch for {path.name}: "
                f"missing={sorted(loaded.missing_keys)[:5]} "
                f"unexpected={sorted(loaded.unexpected_keys)[:5]}")
        if verify_backbone:
            got_digest = backbone_digest(model)
            want = ckpt.get("backbone", {}).get("sha256", "")
            if not want:
                raise RuntimeError(f"checkpoint {path.name} has no backbone hash")
            if got_digest != want:
                raise RuntimeError(
                    "frozen backbone hash mismatch — the model was initialised "
                    f"from different weights than the run used\n  ckpt {want}\n"
                    f"  now  {got_digest}")
    else:
        raise ValueError(f"unknown checkpoint mode {mode!r} in {path}")
    info = {k: v for k, v in ckpt.items()
            if k not in ("trainable", "full", "frozen_keys")}
    return info, ckpt.get("backbone", {})
