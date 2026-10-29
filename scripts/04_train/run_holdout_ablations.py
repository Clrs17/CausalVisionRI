# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Run the 40 inner-only ablation arms for HOLDOUT_MASKED_V1.

The five members come from the frozen intermediate selection lock.  Each
ablation run inherits its member's own resolved ``config.yaml`` and
``checkpoints/initial.pt`` (so batch, weights, temporal recipe, seed and shared
initialization are exactly the member's) and applies only the condition's
model overrides.  Outputs go to a dedicated ablation root, never the main run root.

Conditions (C1,C2,C3s1-3,C4adj,C4sign,C4cf) are fixed here and must match
``scripts/finalize_holdout.py``.

Usage (from the release root):
    python scripts/04_train/run_holdout_ablations.py --selection <selection.json> --dry-run
    python scripts/04_train/run_holdout_ablations.py --selection <selection.json> --jobs 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.protocol import portable_path, sha256_file, sha256_json  # noqa: E402
from wpac_warning.run_integrity import load_completed_marker  # noqa: E402

CONDITIONS = ("C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf")
DEFAULT_OUT = P2_ROOT / "data" / "runs_holdout" / "ablations"
ZERO_ALL = {"model.causal_loss.w_adj_ce": 0,
            "model.causal_loss.w_sign_ate": 0,
            "model.causal_loss.w_counterfactual": 0}


def condition_overrides(condition: str) -> dict:
    if condition == "C1":
        return dict(ZERO_ALL)
    if condition == "C2":
        return {**ZERO_ALL, "model.gnn.enabled": False}
    if condition.startswith("C3s"):
        seed = int(condition[3:])
        if seed <= 0:
            raise ValueError(f"invalid C3 condition {condition!r}")
        return {"model.gnn.prior_shuffle_seed": seed}
    # C4 keeps the member's ORIGINAL weight for the retained term and only
    # zeros the other two; never re-set the retained weight (that would change
    # the original experiment).
    keep = {"C4adj": "w_adj_ce", "C4sign": "w_sign_ate",
            "C4cf": "w_counterfactual"}
    if condition in keep:
        return {f"model.causal_loss.{k}": 0
                for k in ("w_adj_ce", "w_sign_ate", "w_counterfactual")
                if k != keep[condition]}
    raise ValueError(f"unknown ablation condition {condition!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent ablation runs (default 1 = legacy serial)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")

    lock = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if sha256_json(payload) != lock.get("lock_sha256"):
        raise SystemExit("selection lock hash mismatch")
    if lock.get("kind") != "selection":
        raise SystemExit("ablation runs require the intermediate selection lock")
    members = [m["run_id"] for m in lock["members"]]
    if len(members) != 5:
        raise SystemExit(f"expected the Top-5 member set, got {len(members)}")

    runs_root = Path(lock["runs_root"])
    if not runs_root.is_absolute():
        runs_root = (P2_ROOT / runs_root).resolve()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (P2_ROOT / out_root).resolve()
    if out_root == runs_root:
        raise SystemExit("ablation output root must differ from the main run root")

    plan = []
    for member in members:
        member_dir = runs_root / member
        completed = load_completed_marker(member_dir)
        member_cfg = member_dir / "config.yaml"
        if not member_cfg.is_file():
            raise SystemExit(f"member config missing: {member_cfg}")
        initial_pt = member_dir / "checkpoints" / "initial.pt"
        if not initial_pt.is_file():
            raise SystemExit(f"member shared initialization missing: {initial_pt}")
        initial_sha = sha256_file(initial_pt)
        if completed["artifacts"]["checkpoints/initial.pt"] != initial_sha:
            raise SystemExit(f"{member}: initial.pt differs from completed marker")
        for condition in CONDITIONS:
            run_id = f"abl_{condition}_{member}"
            run_dir = out_root / run_id
            if run_dir.exists():
                raise SystemExit(f"ablation run exists: {run_dir} (no implicit "
                                 f"resume; remove it or use a new --out-root)")
            cmd = [sys.executable, "-m", "src.train",
                   "--config", str(member_cfg),
                   "--set", f"train.run_id={run_id}",
                   "--set", f"train.output_root={out_root}",
                   "--set", f"train.init_from={initial_pt}"]
            for key, val in sorted(condition_overrides(condition).items()):
                cmd += ["--set", f"{key}={val}"]
            plan.append({"run_id": run_id, "member": member,
                         "condition": condition, "initial_pt": str(initial_pt),
                         "initial_sha256": initial_sha, "command": cmd})

    manifest = {
        "protocol": "holdout_masked_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_lock": portable_path(Path(args.selection), P2_ROOT),
        "selection_sha256": lock["lock_sha256"],
        "out_root": portable_path(out_root, P2_ROOT),
        "n_runs": len(plan),
        "conditions": list(CONDITIONS),
        "members": members,
        "plan_sha256": sha256_json([p["command"] for p in plan]),
        "jobs": args.jobs,
        "plan": plan,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runner_logs").mkdir(parents=True, exist_ok=True)
    (out_root / "ablation_runner_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    def _run_logged(entry: dict) -> int:
        log_path = out_root / "runner_logs" / f"{entry['run_id']}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("command: " + " ".join(entry["command"]) + "\n")
            log.flush()
            return subprocess.call(entry["command"], cwd=str(P2_ROOT),
                                   stdout=log, stderr=subprocess.STDOUT)

    failed = []
    if args.jobs == 1:
        for p in plan:
            print(f"=== {p['run_id']}", flush=True)
            if _run_logged(p) != 0:
                failed.append(p["run_id"])
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_run_logged, entry): entry
                       for entry in plan}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    returncode = future.result()
                except Exception:
                    returncode = 1
                if returncode != 0:
                    failed.append(entry["run_id"])
                    print(f"=== {entry['run_id']} FAILED "
                          f"(see {out_root / 'runner_logs'})", flush=True)
                else:
                    print(f"=== {entry['run_id']} completed", flush=True)
    if failed:
        raise SystemExit(f"failed ablation runs: {sorted(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
