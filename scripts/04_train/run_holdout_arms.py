# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Run the 22 registered arms under HOLDOUT_MASKED_V1 (inner-only).

Every arm's technical configuration comes ONLY from
``configs/phase2/arms_holdout_v1.yaml`` (via ``src.arms.arm_overrides``).  No metric is
read and nothing here starts test inference.

Refuses to touch an existing run directory without explicit ``--resume``;
write the whole plan first with ``--dry-run``.  ``--jobs 1`` is the legacy
serial path.  Parallel workers use independent runner logs/manifests and never
write the shared root ``runner_manifest.json``.

Usage (from the release root):
    python scripts/04_train/run_holdout_arms.py --jobs 4 --dry-run
    python scripts/04_train/run_holdout_arms.py --arms ens_ep40,ens_v1 --jobs 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.arms import (DEFAULT_REGISTRY, arm_names, arm_overrides,  # noqa: E402
                      load_registry)
from wpac_warning.protocol import portable_path, sha256_file, sha256_json  # noqa: E402
from wpac_warning.utils_cfg import load_cfg                                   # noqa: E402

DEFAULT_CONFIG = P2_ROOT / "configs" / "phase2" / "holdout_masked_v1.yaml"
DEFAULT_OUT = P2_ROOT / "data" / "runs" / "holdout_masked_v1"


def build_command(cfg_path: Path, arm: str, out_root: Path,
                  registry_path: Path, resume: bool = False,
                  log_file: Path | None = None) -> list:
    reg = load_registry(registry_path)
    ov = arm_overrides(reg, arm)
    if "data.population" in ov and str(ov["data.population"]) != "alltc":
        raise ValueError(f"{arm}: holdout arms require population=alltc")
    if "train.inner_only" in ov and not bool(ov["train.inner_only"]):
        raise ValueError(f"{arm}: holdout arms require train.inner_only=true")
    ov["train.inner_only"] = True
    ov["train.output_root"] = str(out_root)
    cmd = [sys.executable, "-m", "src.train", "--config", str(cfg_path)]
    for key in sorted(ov):
        cmd += ["--set", f"{key}={ov[key]}"]
    if resume:
        cmd.append("--resume")
    if log_file is not None:
        cmd += ["--log-file", str(log_file)]
    return cmd


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _run_arm_logged(plan: dict) -> int:
    with open(plan["log"], "w", encoding="utf-8") as log:
        log.write("command: " + " ".join(plan["command"]) + "\n")
        log.flush()
        return subprocess.call(plan["command"], cwd=str(P2_ROOT),
                               stdout=log, stderr=subprocess.STDOUT)


def _run_parallel(manifest: dict, plan: list, out_root: Path, jobs: int) -> None:
    failed = []
    (out_root / "runner_logs").mkdir(parents=True, exist_ok=True)
    for entry in plan:
        record = {
            "protocol": manifest["protocol"],
            "created_at": manifest["created_at"],
            "config_sha256": manifest["config_sha256"],
            "registry_sha256": manifest["registry_sha256"],
            "plan_sha256": manifest["plan_sha256"],
            "jobs": jobs, "arm": entry["arm"], "command": entry["command"],
            "run_dir": entry["run_dir"], "log": entry["log"],
            "train_log": entry["train_log"], "status": "scheduled",
        }
        _write_json_atomic(Path(entry["runner_manifest"]), record)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_arm_logged, entry): entry for entry in plan}
        for future in as_completed(futures):
            entry = futures[future]
            error = None
            try:
                returncode = future.result()
            except Exception as exc:  # aggregate launch/worker failures too
                returncode = None
                error = str(exc)
            record_path = Path(entry["runner_manifest"])
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record.update({
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            if error is not None:
                record["error"] = error
            _write_json_atomic(record_path, record)
            if returncode != 0:
                failed.append(entry["arm"])
                print(f"=== {entry['arm']} FAILED (see {entry['log']})", flush=True)
            else:
                print(f"=== {entry['arm']} completed", flush=True)
    if failed:
        raise SystemExit(f"failed arms: {sorted(failed)}; per-arm logs: "
                         f"{out_root / 'runner_logs'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset (default: all registered)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers; 1 preserves the legacy serial runner")
    ap.add_argument("--resume", action="store_true",
                    help="explicitly pass --resume to each training run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    cfg = load_cfg(args.config)
    if str(cfg.get("data.protocol")) != "holdout_masked_v1":
        raise SystemExit("base config is not a holdout_masked_v1 config")
    population = cfg.get("data.population")
    if population is not None and str(population) != "alltc":
        raise SystemExit("holdout arms require data.population=alltc when provided")
    if not bool(cfg.get("train.inner_only", True)):
        raise SystemExit("holdout arms must run with train.inner_only=true")
    if cfg.get("train.init_from"):
        raise SystemExit("main arms must use fresh seeded initial.pt, not train.init_from")
    reg = load_registry(args.registry)
    names = arm_names(reg)
    if args.arms:
        keep = [a.strip() for a in args.arms.split(",") if a.strip()]
        if len(set(keep)) != len(keep):
            raise SystemExit("--arms contains duplicate arm names")
        unknown = sorted(set(keep) - set(names))
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        names = keep

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (P2_ROOT / out_root).resolve()
    plan = []
    for arm in names:
        run_dir = out_root / arm
        if run_dir.exists() and not args.resume:
            raise SystemExit(f"run directory already exists: {run_dir} — "
                             f"pass --resume explicitly or use a new --out-root "
                             f"(no implicit resume)")
        train_log = (out_root / "runner_logs" / f"{arm}.train.log"
                     if args.jobs > 1 else None)
        cmd = build_command(Path(args.config), arm, out_root,
                            Path(args.registry), resume=args.resume,
                            log_file=train_log)
        entry = {"arm": arm, "command": cmd, "run_dir": str(run_dir)}
        if args.jobs > 1:
            entry["log"] = str(out_root / "runner_logs" / f"{arm}.log")
            entry["train_log"] = str(train_log)
            entry["runner_manifest"] = str(
                out_root / "runner_manifests" / f"{arm}.json")
        plan.append(entry)

    manifest = {
        "protocol": "holdout_masked_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "registry": str(Path(args.registry).resolve()),
        "registry_sha256": sha256_file(args.registry),
        "out_root": portable_path(out_root, P2_ROOT),
        "n_arms": len(plan),
        "plan_sha256": sha256_json([p["command"] for p in plan]),
        "arms": plan,
    }
    if args.jobs > 1:
        manifest["jobs"] = args.jobs
        manifest["parallel"] = True
        manifest["schedule"] = {
            "max_concurrent": min(args.jobs, len(plan)),
            "arms": list(names),
            "runner_logs": str(out_root / "runner_logs"),
            "runner_manifests": str(out_root / "runner_manifests"),
        }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if args.jobs > 1:
        _run_parallel(manifest, plan, out_root, args.jobs)
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runner_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    failed = []
    for p in plan:
        print(f"=== {p['arm']}", flush=True)
        rc = subprocess.call(p["command"], cwd=str(P2_ROOT))
        if rc != 0:
            failed.append(p["arm"])
    if failed:
        raise SystemExit(f"failed arms: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
