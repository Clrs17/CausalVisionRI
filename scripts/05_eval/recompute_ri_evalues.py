# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Recompute the RI-row E-values in ``ate_results.csv`` with the true panel
base rate instead of the hard-coded 0.113.

Background
    ``wpac_causal_phase1/src/effect_estimation.py`` computes the binary-outcome
    E-value (VanderWeele & Ding 2017) through

        _e_value(ate, outcome_std, binary, base_rate=1014 / 12647)

    i.e. OR = exp(ate) -> RR = OR / (1 - p0 + p0 * OR) with p0 = base_rate.
    The RI prevalence of the *fit* panel is 1014 / 12647 = 0.080177;
    RI E-values from earlier pipeline output used the incorrect value 0.113.

What this script does
    Reads the existing ``ate_results.csv``, re-runs the two E-value helpers
    taken **verbatim from the source file** (extracted with ``ast`` and
    ``exec``-ed, so the formula cannot drift from the implementation) with
    ``base_rate = BASE_RATE`` for rows whose ``outcome == "RI"``, and writes
    the CSV back.  Only ``e_value`` and ``e_value_ci_limit`` on RI rows change;
    row order, column order, every other field and the file layout are
    preserved byte-for-byte (verified before writing).

    Rows with a continuous outcome (DV24) do not use ``base_rate`` at all and
    are left untouched.

Usage
    python scripts/05_eval/recompute_ri_evalues.py --dry-run
    python scripts/05_eval/recompute_ri_evalues.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "src" / "wpac_causal" / "effect_estimation.py"
DEFAULT_CSV = (
    REPO / "data" / "phase1" / "processed_masked" / "holdout_fit0017"
    / "prior" / "ate_results.csv"
)
DEFAULT_PANEL = (
    REPO / "data" / "phase1" / "processed_masked" / "holdout_fit0017"
    / "prior" / "fit_prior_panel.parquet"
)

# RI prevalence of the fit panel: 1014 / 12647.
BASE_RATE = 1014.0 / 12647.0
# Sanity targets from the independent K3 recalculation.
EXPECTED = {"OHC_OA": 3.0260, "SHRD": 3.3863}
TARGET_COLS = ("e_value", "e_value_ci_limit")


def load_source_helpers(path: Path):
    """Exec ``_e_value`` / ``_e_value_ci_limit`` straight from the source file."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = ("_e_value", "_e_value_ci_limit")
    lines = src.splitlines()
    segs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segs[node.name] = "\n".join(lines[node.lineno - 1:node.end_lineno])
    missing = [n for n in wanted if n not in segs]
    if missing:
        raise SystemExit(f"could not find {missing} in {path}")

    ns = {"np": np}
    exec("\n\n".join(segs[n] for n in wanted), ns)  # noqa: S102 - provenance by construction
    ev, ev_ci = ns["_e_value"], ns["_e_value_ci_limit"]

    default = ev.__defaults__[-1] if ev.__defaults__ else None
    if default != 0.08017711710287025:
        raise SystemExit(f"unexpected default base_rate {default!r} in {path}")
    return ev, ev_ci, segs


def panel_ri_rate(panel_path: Path) -> tuple[float, int, int, float]:
    df = pd.read_parquet(panel_path)
    ri = df["RI"]
    n = int(ri.notna().sum())
    k = int(np.nansum(ri.values))
    return float(np.nanmean(ri.values)), k, n, float(df["DV24"].std())


def detect_newline(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def serialize(rows: list[list[str]], newline: str) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=newline)
    w.writerows(rows)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    for p in (args.csv, args.source, args.panel):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    ev, ev_ci, segs = load_source_helpers(args.source)
    rate, k, n, dv24_std = panel_ri_rate(args.panel)
    print("== panel RI base rate ==")
    print(f"panel  : {args.panel}")
    print(f"RI sum / n = {k} / {n} = {k / n!r}")
    print(f"panel mean = {rate!r}   (script constant {BASE_RATE!r})")
    if abs(rate - BASE_RATE) > 1e-9:
        raise SystemExit(f"panel RI rate {rate!r} != BASE_RATE {BASE_RATE!r}")
    print(f"DV24 std (continuous branch only, unused for RI) = {dv24_std!r}")

    raw = args.csv.read_bytes().decode("utf-8")
    newline = detect_newline(raw)
    rows = list(csv.reader(io.StringIO(raw, newline="")))
    header = rows[0]
    col = {c: i for i, c in enumerate(header)}
    for c in ("treatment", "outcome", "ate", *TARGET_COLS):
        if c not in col:
            raise SystemExit(f"column {c!r} not in {args.csv}")
    i_t, i_o, i_ate = col["treatment"], col["outcome"], col["ate"]
    i_ev, i_ci = col["e_value"], col["e_value_ci_limit"]
    i_blo, i_bhi = col["ci_boot_low"], col["ci_boot_high"]
    i_lo, i_hi = col["ci_low"], col["ci_high"]

    # Format is preserved only if re-serializing the untouched rows reproduces
    # the original file exactly; refuse to write otherwise.
    if serialize(rows, newline) != raw:
        raise SystemExit("CSV round-trip is not byte-identical; refusing to write")

    new_rows = [list(r) for r in rows]
    changes = []
    for r, out_row in zip(rows[1:], new_rows[1:]):
        if r[i_o] != "RI":
            continue
        ate = float(r[i_ate])
        blo, bhi = r[i_blo], r[i_bhi]
        if blo and bhi:
            lo, hi = float(blo), float(bhi)
        else:
            lo, hi = float(r[i_lo]), float(r[i_hi])
        old_ev, old_ci = r[i_ev], r[i_ci]
        new_ev = repr(ev(ate, dv24_std, True, BASE_RATE))
        new_ci = repr(ev_ci(ate, lo, hi, dv24_std, True, BASE_RATE))
        out_row[i_ev], out_row[i_ci] = new_ev, new_ci
        changes.append((r[i_t], old_ev, new_ev, old_ci, new_ci))

    new_text = serialize(new_rows, newline)

    # Post-check: every field except the two targets on RI rows is unchanged.
    check = list(csv.reader(io.StringIO(new_text, newline="")))
    if len(check) != len(rows):
        raise SystemExit("row count changed")
    if check[0] != header:
        raise SystemExit("header changed")
    for old_r, new_r in zip(rows[1:], check[1:]):
        if len(old_r) != len(new_r):
            raise SystemExit("field count changed")
        for j, (a, b) in enumerate(zip(old_r, new_r)):
            if a == b:
                continue
            if old_r[i_o] != "RI":
                raise SystemExit(f"non-RI row touched: {old_r[i_t]}")
            if j not in (i_ev, i_ci):
                raise SystemExit(f"unexpected column change at {header[j]} ({old_r[i_t]})")

    print("\n== formula (verbatim from source) ==")
    print(segs["_e_value"])
    print()
    print(segs["_e_value_ci_limit"])

    print("\n== before / after (RI rows only) ==")
    print(f"{'treatment':<10} {'e_value':>9} -> {'new':>9}   {'ci_limit':>9} -> {'new':>9}")
    for t, o_ev, n_ev, o_ci, n_ci in changes:
        print(f"{t:<10} {float(o_ev):9.4f} -> {float(n_ev):9.4f}   "
              f"{float(o_ci):9.4f} -> {float(n_ci):9.4f}")
    for t, o_ev, n_ev, o_ci, n_ci in changes:
        if t in EXPECTED:
            print(f"  check {t:<7} new e_value {float(n_ev):.4f} "
                  f"(K3 expected ~{EXPECTED[t]:.4f}, diff={float(n_ev) - EXPECTED[t]:+.5f})")

    n_changed = sum(1 for a, b in zip(rows, check) if a != b)
    print(f"\nrows changed: {n_changed} RI rows; "
          f"{(len(rows) - 1) - len(changes)} non-RI rows byte-identical")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    backup = args.csv.with_name(args.csv.name + ".bak_base_rate")
    if backup.exists():
        print(f"\nbackup already exists, keeping it: {backup}")
    else:
        shutil.copy2(args.csv, backup)
        print(f"\nbackup written: {backup}")
    args.csv.write_bytes(new_text.encode("utf-8"))
    print(f"updated: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
