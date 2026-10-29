# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Frozen 22-arm registry for HOLDOUT_MASKED_V1.

The registry is the only source of arm identity: architecture, weights role,
fine-tuning mode, temporal recipe, batch and the exact config overrides.  It
was reconstructed from the historical runners and training logs and contains
no historical metric.

``holdout_select`` requires the discovered run set to equal the registry arms
(22 by default) and takes the family assignment and tie-break order from here.

CLI (prints the ``--set`` list for one arm; does not train):

    python -m src.arms --registry config/arms_holdout_v1.yaml --arm ens_ep40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

P2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = P2_ROOT / "configs" / "phase2" / "arms_holdout_v1.yaml"
FAMILIES = ("lingbot_sf", "lingbot_mf", "mae_sf", "mae_mf")


class RegistryError(ValueError):
    pass


def load_registry(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_REGISTRY
    if not p.is_file():
        raise FileNotFoundError(f"arm registry not found: {p}")
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))
    reg["_path"] = str(p.resolve())
    validate_registry(reg)
    return reg


def validate_registry(reg: Dict[str, Any]) -> None:
    arms = reg.get("arms")
    if not isinstance(arms, list) or not arms:
        raise RegistryError("registry has no arms")
    names = [a.get("arm") for a in arms]
    if len(set(names)) != len(names):
        raise RegistryError(f"duplicate arm names: {names}")
    n_expected = int(reg.get("n_arms", len(arms)))
    if len(arms) != n_expected:
        raise RegistryError(f"registry declares n_arms={n_expected} but lists "
                            f"{len(arms)}")
    family_order = reg.get("family_order", list(FAMILIES))
    if set(family_order) != set(FAMILIES):
        raise RegistryError(f"family_order must cover {FAMILIES}")
    weights = reg.get("assets", {}).get("weights", {})
    embeds = reg.get("assets", {}).get("frame_embeds", {})
    for a in arms:
        name = a["arm"]
        for key in ("family", "backbone", "weights_role", "ft_mode", "batch"):
            if key not in a:
                raise RegistryError(f"{name}: missing '{key}'")
        if a["family"] not in family_order:
            raise RegistryError(f"{name}: unknown family {a['family']!r}")
        if a["backbone"] not in ("vit_mae", "lingbot"):
            raise RegistryError(f"{name}: unknown backbone {a['backbone']!r}")
        if a["weights_role"] not in weights:
            raise RegistryError(f"{name}: weights_role "
                                f"{a['weights_role']!r} not in assets.weights")
        w = weights[a["weights_role"]]
        if w.get("kind") != a["backbone"]:
            raise RegistryError(f"{name}: weights_role kind {w.get('kind')!r} "
                                f"!= backbone {a['backbone']!r}")
        if int(a["batch"]) <= 0:
            raise RegistryError(f"{name}: batch must be positive")
        temporal = list(a.get("temporal") or [])
        if any(int(lag) > 0 or int(lag) % 6 != 0 for lag in temporal):
            raise RegistryError(f"{name}: temporal lags must be <=0 and "
                                f"multiples of 6: {temporal}")
        er = a.get("embeds_role")
        if er:
            if er not in embeds:
                raise RegistryError(f"{name}: embeds_role {er!r} not in "
                                    f"assets.frame_embeds")
            if not temporal:
                raise RegistryError(f"{name}: embeds_role requires temporal lags")
            if a["ft_mode"] != "frozen":
                raise RegistryError(f"{name}: embedding caches are only valid "
                                    f"for frozen backbones")
            if embeds[er].get("from_weights_role") != a["weights_role"]:
                raise RegistryError(
                    f"{name}: embeds cache {er!r} was built from "
                    f"{embeds[er].get('from_weights_role')!r}, arm uses "
                    f"{a['weights_role']!r}")


def arm_names(reg: Dict[str, Any]) -> List[str]:
    return [a["arm"] for a in reg["arms"]]


def arm_family(reg: Dict[str, Any], arm: str) -> str:
    for a in reg["arms"]:
        if a["arm"] == arm:
            return a["family"]
    raise RegistryError(f"arm {arm!r} not in registry")


def arm_entry(reg: Dict[str, Any], arm: str) -> Dict[str, Any]:
    for a in reg["arms"]:
        if a["arm"] == arm:
            return a
    raise RegistryError(f"arm {arm!r} not in registry")


def arm_overrides(reg: Dict[str, Any], arm: str,
                  config_path: str | Path | None = None) -> Dict[str, Any]:
    """Resolve one arm into a flat {dot.path: value} override dict.

    Role paths (weights, embedding cache) are substituted from
    ``assets``; temporal lags are expanded into BOTH the dataset and model
    keys so they cannot drift apart.
    """
    entry = arm_entry(reg, arm)
    out: Dict[str, Any] = {"train.run_id": arm,
                           "train.batch_size": int(entry["batch"])}
    out.update({str(k): v for k, v in (entry.get("overrides") or {}).items()})

    weights = reg["assets"]["weights"][entry["weights_role"]]
    backend = entry["backbone"]
    out["model.vision.backend"] = backend
    if backend == "vit_mae":
        out["model.vision.mae_checkpoint"] = weights["path"]
        out["model.vision.freeze_backbone"] = entry["ft_mode"] == "frozen"
    else:
        out["model.vision.lingbot_dir"] = weights["path"]
        out["model.vision.lingbot_variant"] = weights.get("variant", "small")
        if entry["ft_mode"] == "frozen":
            out["model.vision.ft_mode"] = "frozen"
            out["model.vision.freeze_backbone"] = True
        else:
            out["model.vision.freeze_backbone"] = False

    temporal = list(entry.get("temporal") or [])
    if temporal:
        out["model.vision.temporal_lags"] = temporal
        out["data.ir_lag_steps"] = temporal
    er = entry.get("embeds_role")
    if er:
        out["data.ir_embeds_dir"] = reg["assets"]["frame_embeds"][er]["path"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print one arm's --set list")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "phase2" /
                                            "holdout_masked_v1.yaml"),
                    help="base config (recorded for provenance only)")
    args = ap.parse_args(argv)
    reg = load_registry(args.registry)
    ov = arm_overrides(reg, args.arm)
    print(f"# config: {args.config}")
    for k in sorted(ov):
        v = ov[k]
        print(f"--set {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
