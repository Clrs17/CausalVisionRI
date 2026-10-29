# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
#!/usr/bin/env python
"""Generate the supplementary tables S1-S9 as LaTeX (figure-publication style).

Every number is read from a frozen artefact; nothing is trained and no new
inference is performed.  The script also cross-checks the generated numbers
against the values printed in docs/RESULTS.md and
docs/SHIPS12_BASELINE.md, and against internal invariants (row counts, CI
bracketing, Holm-significance counts, Jaccard/label-agreement algebra), then
prints a per-table row count plus every check.

The output is a standalone LaTeX body: \\input it into whatever manuscript
source you keep outside this repository; it needs \\usepackage{booktabs} and
\\usepackage{longtable} in that document's preamble.

Usage:
    python scripts/05_eval/make_sm_tables.py [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "data" / "phase1"
P2 = ROOT

ATE_CSV = P1 / "processed_masked/holdout_fit0017/prior/ate_results.csv"
EVAL_DIR = P2 / "data/analysis/eval"
SELECTION_CSV = P2 / "protocol_locks/selection.csv"
SELECTION_JSON = P2 / "protocol_locks/selection.json"
COMPARISON_LOCK = P2 / "protocol_locks/comparison_secondary.json"
GATE_CSV = P2 / "data/analysis/gate_participation.csv"
COVERAGE_CSV = P2 / "data/analysis/compare/coverage.csv"
WATERFALL_CSV = P2 / "data/analysis/compare/waterfall.csv"
SHIPS12_MD = P2 / "docs/SHIPS12_BASELINE.md"
RESULTS_MD = P2 / "docs/RESULTS.md"
BASELINE_COMMON_PY = P2 / "src" / "wpac_warning" / "baseline_common.py"
OUT_DEFAULT = P2 / "data" / "analysis" / "sm_tables.tex"

# Alpha levels actually used by the frozen artefacts (see effect_estimation.py:
# bootstrap percentile CI at 2.5/97.5; phase-2 eval JSONs: alpha = 0.10).
ATE_CI_LEVEL = "95"
ABL_CI_LEVEL = "90"


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}


def esc(text) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def is_num(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt(x, nd: int = 3) -> str:
    """Fixed-point number with a real minus sign (math mode)."""
    if not is_num(x):
        return "--"
    v = float(x)
    if abs(v) < 0.5 * 10.0 ** (-nd):  # never print '-0.000'
        v = 0.0
    s = f"{abs(v):.{nd}f}"
    return f"$-${s}" if v < 0 else s


def pfmt(p) -> str:
    if not is_num(p):
        return "--"
    p = float(p)
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def count(n) -> str:
    return f"{int(n):,}"


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing input: {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {rel(path)}")
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh))


def md_table(path: Path, heading: str) -> list[list[str]]:
    """Rows of the first markdown table under a heading containing `heading`."""
    if not path.exists():
        raise SystemExit(f"missing input: {rel(path)}")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("#") and heading in ln), None)
    if start is None:
        raise SystemExit(f"{rel(path)}: heading {heading!r} not found")
    rows: list[list[str]] = []
    for ln in lines[start + 1:]:
        if ln.startswith("#"):
            break
        if ln.strip().startswith("|"):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif rows:
            break
    if len(rows) < 3:
        raise SystemExit(f"{rel(path)}: no table rows under {heading!r}")
    return rows[2:]  # drop header + |---| separator


def strip_md(cell: str) -> str:
    return re.sub(r"\*\*", "", cell).replace("`", "").strip()


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
CHECKS: list[tuple[str, str, bool, str]] = []
CAVEATS: list[str] = []


def check(table: str, name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((table, name, bool(ok), detail))


def check_map(table: str, name: str, actual: dict, expected: dict, tol: float) -> None:
    bad = []
    for key, want in expected.items():
        got = actual.get(key)
        if not is_num(got) or abs(float(got) - float(want)) > tol:
            bad.append(f"{key}: data={got if got is None else round(float(got), 4)}"
                       f" doc={want}")
    check(table, name, not bad,
          f"{len(expected)} values within {tol:g}"
          if not bad else f"{len(bad)} mismatch(es) -> " + "; ".join(bad[:6]))


# --------------------------------------------------------------------------- #
# LaTeX emitters
# --------------------------------------------------------------------------- #
def _body_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def _spec_width(colspec: str) -> int:
    s = re.sub(r"@\{[^}]*\}", "", colspec).replace(" ", "")
    s = re.sub(r">\{[^}]*\}", "", s)
    s = re.sub(r"[pmb]\{[^}]*\}", "X", s)
    s = s.replace("|", "")
    return len(s)


def _header_row(header: list[str]) -> str:
    cells = [h if h.startswith("$") else rf"\textbf{{{h}}}" for h in header]
    return _body_row(cells)


def _guard(label: str, colspec: str, header: list[str], rows: list[str]) -> int:
    ncol = len(header)
    if _spec_width(colspec) != ncol:
        raise ValueError(f"table {label}: colspec has {_spec_width(colspec)} columns "
                         f"but the header has {ncol}")
    for row in rows:
        if row.lstrip().startswith(r"\multicolumn"):
            continue
        if row.count("&") + 1 != ncol:
            raise ValueError(f"table {label}: row has {row.count('&') + 1} cells "
                             f"(expected {ncol}): {row[:60]}")
    return ncol


def longtable(label, caption, colspec, header, rows, note=None, source=None,
              tabcolsep=None) -> str:
    ncol = _guard(label, colspec, header, rows)
    hdr = _header_row(header)
    out = [r"{\scriptsize"]
    if tabcolsep:
        out.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}")
    out += [r"\begin{longtable}{" + colspec + "}",
            r"\caption{" + caption + "}", r"\label{tab:" + label + r"}\\",
            r"\toprule", hdr, r"\midrule", r"\endfirsthead",
            r"\multicolumn{" + str(ncol) + r"}{@{}l}{\footnotesize\itshape (continued)}\\",
            r"\toprule", hdr, r"\midrule", r"\endhead",
            r"\bottomrule", r"\endfoot"]
    out += rows
    out += [r"\end{longtable}}"]
    if note:
        out += [r"\par\smallskip\noindent{\footnotesize\raggedright " + note + r"\par}"]
    if source:
        out.append("% source: " + source)
    return "\n".join(out)


def table(label, caption, colspec, header, rows, note=None, source=None,
          placement="htbp", tabcolsep=None) -> str:
    ncol = _guard(label, colspec, header, rows)
    out = [r"\begin{table}[" + placement + "]", r"\centering", r"\scriptsize"]
    if tabcolsep:
        out.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}")
    out += [r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
            r"\begin{tabular}{" + colspec + "}", r"\toprule",
            _header_row(header), r"\midrule"]
    out += rows
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    if note:
        out.append(r"\par\smallskip\noindent{\footnotesize\raggedright " + note + r"\par}")
    out.append(r"\end{table}")
    if source:
        out.append("% source: " + source)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# S1 - full ATE table
# --------------------------------------------------------------------------- #
S1_TREATMENTS = ["SHRD", "RHLO", "VORT", "PV", "D200", "WARM_CORE", "MFC",
                 "W500", "OHC_IC", "OHC_OA"]
S1_TREATMENT_NAMES = {
    "SHRD": "850--200 hPa vertical wind shear",
    "RHLO": "low-level relative humidity",
    "VORT": "850 hPa relative vorticity",
    "PV": "potential vorticity",
    "D200": "200 hPa divergence",
    "WARM_CORE": "warm-core 200-hPa temperature anomaly (inner minus outer)",
    "MFC": "moisture flux convergence",
    "W500": "500 hPa pressure velocity (negative values denote ascent)",
    "OHC_IC": "inner-core ocean heat content",
    "OHC_OA": "outer-area ocean heat content",
}


def build_s1() -> tuple[str, int]:
    rows = read_csv(ATE_CSV)
    by_key = {(r["treatment"], r["outcome"]): r for r in rows}
    expect = {(t, o) for t in S1_TREATMENTS for o in ("DV24", "RI")}
    check("S1", "table is complete (10 treatments x 2 outcomes)", set(by_key) == expect,
          f"{len(by_key)} rows")

    body, indeterminate, unbracketed, ev_bad, ev_missing = [], [], [], [], []
    for t in S1_TREATMENTS:
        for o in ("DV24", "RI"):
            r = by_key[(t, o)]
            ate, lo, hi = (float(r["ate_boot"]), float(r["ci_boot_low"]),
                           float(r["ci_boot_high"]))
            if not (lo <= ate <= hi):
                unbracketed.append(f"{t}/{o}")
            if lo <= 0.0 <= hi:
                indeterminate.append(f"{t}/{o}")
            ev, ev_ci = r.get("e_value"), r.get("e_value_ci_limit")
            if not (is_num(ev) and is_num(ev_ci)):
                ev_missing.append(f"{t}/{o}")
            elif float(ev_ci) > float(ev) + 1e-9 or float(ev) < 1.0:
                ev_bad.append(f"{t}/{o}: {float(ev):.3f}/{float(ev_ci):.3f}")
            passed = f"{int(r['refutation_passed'])}/{int(r['refutation_total'])}"
            adj = [a.strip() for a in (r["adjustment_set"] or "").split(",")
                   if a.strip() and a.strip() != "(empty)"]
            body.append(_body_row([esc(t), esc(o), fmt(ate), fmt(lo), fmt(hi),
                                   fmt(ev), fmt(ev_ci),
                                   pfmt(r["p_value_boot"]), passed,
                                   esc(", ".join(adj)) if adj else "none"]))

    check("S1", "bootstrap CI brackets the point estimate (internal)", not unbracketed,
          f"{len(body)} rows" if not unbracketed else "; ".join(unbracketed))
    check("S1", "only three contrasts have a CI crossing zero",
          len(indeterminate) == 3, ", ".join(indeterminate) or "none crossed zero")
    check("S1", "DV24 shear effect is negative and strongly non-null",
          float(by_key[("SHRD", "DV24")]["ate_boot"]) < -4.0
          and float(by_key[("SHRD", "DV24")]["ci_boot_high"]) < 0.0,
          f"ATE={float(by_key[('SHRD', 'DV24')]['ate_boot']):.3f}")
    n_obs = {int(r["n_obs"]) for r in rows}
    check("S1", "one common estimation sample for all twenty contrasts",
          len(n_obs) == 1, f"n_obs = {sorted(n_obs)}")
    check("S1", "E-values present for all twenty contrasts", not ev_missing,
          "; ".join(ev_missing) or "20/20 rows")
    check("S1", "E-value >= 1 and its CI limit never exceeds the point E-value", not ev_bad,
          "; ".join(ev_bad) or "20/20 consistent")
    null_ev = sorted(f"{t}/{o}" for t in S1_TREATMENTS for o in ("DV24", "RI")
                     if abs(float(by_key[(t, o)]["e_value_ci_limit"]) - 1.0) < 1e-9)
    check("S1", "E-value CI limit equals 1 exactly where the CI reaches the null",
          null_ev == sorted(indeterminate),
          ", ".join(null_ev) or "none")

    caption = (
        "Ocean heat content provides the largest positive graph-adjusted effects on "
        "24-h intensity change, and only three of the twenty contrasts have a bootstrap "
        "confidence interval that crosses zero. Graph-adjusted effect estimates "
        "for the ten environmental treatments on 24-h intensity change (DV24, kt) and "
        "rapid intensification (RI, conditional log-odds) in the fit-period (2000--2017) "
        f"panel (n~=~{count(max(n_obs))}), with a " + ATE_CI_LEVEL
        + r"\% percentile bootstrap interval (200 storm-level resamples), "
        "the E-value at the estimate and at the confidence limit nearest the null, the "
        "bootstrap p-value, the number of the four refutation tests passed, and the "
        "backdoor adjustment set implied by the locked DAG (``none'' = empty set)."
    )
    names = "; ".join(f"{esc(k)}, {esc(v)}" for k, v in S1_TREATMENT_NAMES.items())
    note = ("DV24 ($\\Delta V_{24}$), 24-h intensity change; RI, rapid intensification; ATE, average "
            "treatment effect. Effects are estimated on the fit period (2000--2017) only; "
            "bootstrap intervals are conditional on the locked identification DAG. "
            "E-value is the minimum strength of "
            "association that an unmeasured confounder would need to have with both "
            "treatment and outcome to explain the effect away; the E-value CI limit is the "
            "same quantity evaluated at the confidence limit closest to zero, and equals "
            "1.000 when that limit already reaches the null. Refut., number of "
            "refutation tests passed (four, or three where the adjustment set is "
            "empty); these tests use LPM-scale reference effects for both outcomes. "
            "Treatments: " + names + ". "
            "D200 is the only treatment without a directed path to DV24 in the identification DAG, and its RI row "
            "relies on an inserted structural edge; both values are reported as associations "
            "rather than identified ATEs. D200 is excluded from the trusted signs by a "
            "protocol-locked rule. RI estimates are conditional log-odds per SD (binomial GLM), "
            "not a marginal ATE.")
    tex = longtable("S1", caption, r"@{}lrrrrrrrrp{3.1cm}@{}",
                    ["Treatment", "Outcome", "Effect", "CI low", "CI high", "E-value",
                     "E-value CI limit", r"$\boldsymbol{p}_{\mathrm{boot}}$",
                     "Refut.", "Adjustment set"],
                    body, note, rel(ATE_CSV), tabcolsep=2.5)
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S3 - per-year metrics, all models
# --------------------------------------------------------------------------- #
S2_MODELS = [
    ("main_2024", "main", "CVRI-Ens (Top-5)"),
    ("baseline_ships12_lda_cohc", "cohc", "SHIPS-12 LDA (COHC)"),
    ("baseline_ships12_lda", "ohcd", "SHIPS-12 LDA (OHCD)"),
    ("baseline_lda", "lda", "LDA (21 panel predictors)"),
    ("baseline_lr", "lr", "LR (uncalibrated)"),
]
S2_YEARS = [2020, 2021, 2022, 2023, 2024]
S2_METRICS = ["BSS_dev", "POD", "FARate", "FARatio", "CSI"]
S2_HEADERS = ["BSS$_{\\mathrm{dev}}$", "POD", "POFD", "FARatio", "CSI"]

# RESULTS Sec. 2 (main, per year) as printed in the document.
DOC_S2_MAIN = {
    (2020, "BSS_dev"): 0.2765, (2020, "POD"): 0.674, (2020, "FARate"): 0.097,
    (2020, "FARatio"): 0.633, (2020, "CSI"): 0.312,
    (2021, "BSS_dev"): 0.2980, (2021, "POD"): 0.659, (2021, "FARate"): 0.073,
    (2021, "FARatio"): 0.642, (2021, "CSI"): 0.302,
    (2022, "BSS_dev"): 0.1592, (2022, "POD"): 0.639, (2022, "FARate"): 0.099,
    (2022, "FARatio"): 0.697, (2022, "CSI"): 0.258,
    (2023, "BSS_dev"): 0.2868, (2023, "POD"): 0.660, (2023, "FARate"): 0.076,
    (2023, "FARatio"): 0.545, (2023, "CSI"): 0.368,
    (2024, "BSS_dev"): 0.2520, (2024, "POD"): 0.761, (2024, "FARate"): 0.142,
    (2024, "FARatio"): 0.653, (2024, "CSI"): 0.313,
}
# RESULTS Sec. 7 baseline table: BSS_dev / CSI / POD / POFD.
DOC_S7_BASELINES = {
    ("ohcd", 2020): [0.201, 0.329, 0.581, 0.064],
    ("ohcd", 2021): [0.112, 0.230, 0.659, 0.115],
    ("cohc", 2020): [0.200, 0.281, 0.419, 0.041],
    ("cohc", 2021): [0.134, 0.236, 0.477, 0.063],
    ("lda", 2020): [0.220, 0.193, 0.744, 0.238],
    ("lda", 2021): [0.096, 0.161, 0.818, 0.251],
    ("lr", 2020): [-2.028, 0.240, 0.721, 0.167],
    ("lr", 2021): [-2.708, 0.178, 0.682, 0.175],
}
# RESULTS Sec. 3 pooled rows (BSS_dev).
DOC_S3_POOLED = {"main": 0.2590, "cohc": 0.1048, "ohcd": 0.0947,
                 "lda": 0.0976, "lr": -2.0742}


def build_s2() -> tuple[str, int]:
    data, body, n_all, year_n = {}, [], {}, {}
    dev_rates, dev_basis = set(), set()
    dev_years = None
    n_data = 0
    for fname, short, label in S2_MODELS:
        d = read_json(EVAL_DIR / f"{fname}.json")
        metrics = d["metrics"]
        n_all[short] = int(metrics["all"]["n"])
        year_n[short] = [int(metrics[f"year{y}"]["n"]) for y in S2_YEARS]
        dev = d.get("dev_rate") or {}
        if dev:
            dev_rates.add(round(float(dev["dev_rate"]), 6))
            dev_basis.add(str(dev.get("bss_primary", "")))
            years = (int(dev["fit_years"][0]), int(dev["inner_years"][-1]))
            dev_years = years if dev_years is None else dev_years
        body.append(r"\multicolumn{7}{@{}l}{\textit{" + esc(label) + r"}}\\")
        for year in S2_YEARS + ["all"]:
            key = "all" if year == "all" else f"year{year}"
            m = metrics[key]
            cells = [str(year) if year != "all" else "2020--24", count(m["n"])]
            for metric in S2_METRICS:
                data[(short, year, metric)] = float(m[metric])
                cells.append(fmt(m[metric]))
            if year == "all":
                cells = [rf"\textbf{{{cells[0]}}}"] + [rf"\textbf{{{c}}}" for c in cells[1:]]
            body.append(_body_row(cells))
            n_data += 1

    # every model must cover the same pooled test set
    check("S2", "all five models evaluate the same 3,004 rows",
          len(set(n_all.values())) == 1 and n_all["main"] == 3004,
          f"n = {sorted(set(n_all.values()))}; 5 models x 5 years + pooled = 30 rows")
    check("S2", "per-year row counts sum to the pooled count",
          all(sum(year_n[s]) == n_all[s] for s in n_all),
          f"sum({[sum(year_n['main'])]}) = {n_all['main']}")

    main_map = {(y, k): v for (y, k), v in DOC_S2_MAIN.items()}
    actual_main = {(y, k): data[("main", y, k)] for y in S2_YEARS for k in S2_METRICS}
    check_map("S2", "CVRI-Ens per-year metrics match RESULTS Sec. 2", actual_main,
              main_map, 6e-4)
    # Sec. 7 restates 2020 and 2021 from the same frozen predictions
    s7_main = {k: v for k, v in main_map.items() if k[0] in (2020, 2021)}
    check_map("S2", "CVRI-Ens 2020-2021 subset restated in RESULTS Sec. 7",
              {k: v for k, v in actual_main.items() if k[0] in (2020, 2021)},
              s7_main, 6e-4)

    s7_rows = sorted({"BSS_dev": 0, "CSI": 1, "POD": 2, "FARate": 3}.items(),
                     key=lambda kv: kv[1])
    actual_s7 = {}
    for (short, year), want in DOC_S7_BASELINES.items():
        for metric, idx in s7_rows:
            actual_s7[(short, year, metric)] = data[(short, year, metric)]
    expected_s7 = {(short, year, metric): want[idx]
                   for (short, year), want in DOC_S7_BASELINES.items()
                   for metric, idx in s7_rows}
    check_map("S2", "baseline 2020-2021 rows match RESULTS Sec. 7", actual_s7,
              expected_s7, 6e-4)
    pooled = {s: data[(s, "all", "BSS_dev")] for s in DOC_S3_POOLED}
    check_map("S2", "pooled rows match RESULTS Sec. 3", pooled, DOC_S3_POOLED, 6e-4)

    # the headline ordering claim used in the caption
    others = [m[1] for m in S2_MODELS[1:]]
    best_bss = all(data[("main", y, "BSS_dev")] > max(
        data[(s, y, "BSS_dev")] for s in others) for y in S2_YEARS)
    csi_winner = {y: max(["main"] + others, key=lambda s: data[(s, y, "CSI")])
                  for y in S2_YEARS}
    csi_main = [y for y in S2_YEARS if csi_winner[y] == "main"]
    check("S2", "caption claim: ensemble best BSS_dev in 5/5 years, best CSI in 4/5",
          best_bss and len(csi_main) == 4 and csi_winner[2020] == "ohcd",
          f"BSS 5/5; CSI best in {csi_main}, 2020 goes to {csi_winner[2020]}")
    ships_pos = all(data[(s, y, "BSS_dev")] > 0 for s in ("cohc", "ohcd") for y in S2_YEARS)
    check("S2", "caption claim: both SHIPS-12 variants positively skilled every year",
          ships_pos, "12/12 year-cells positive")
    lr_neg = all(data[("lr", y, "BSS_dev")] < 0 for y in S2_YEARS)
    check("S2", "note claim: LR baseline BSS_dev is negative in every year",
          lr_neg, "5/5 year-cells negative")
    neg_cells = [f"{s}/{y}" for s in others if s != "lr" for y in S2_YEARS
                 if data[(s, y, "BSS_dev")] < 0]
    check("S2", "note claim: outside the LR baseline, LDA 2022 is the only negative cell",
          neg_cells == ["lda/2022"], ", ".join(neg_cells))
    check("S2", "all five models share one fixed development climatology",
          len(dev_rates) == 1 and dev_years == (2000, 2019)
          and all("2000-2019" in b for b in dev_basis if b),
          f"rate={sorted(dev_rates)}, years={dev_years}, basis={sorted(b for b in dev_basis if b)}")
    dev_rate = min(dev_rates) if dev_rates else 0.0
    dev_txt = f"{dev_rate:.4f}"
    lda_2022 = fmt(data[("lda", 2022, "BSS_dev")])

    caption = (
        "CVRI-Ens has the highest BSS$_{\\mathrm{dev}}$ in every one of the five test years and "
        "the highest CSI in four of them (SHIPS-12 OHCD leads on CSI in 2020 alone), and "
        "both SHIPS-12 variants stay positively skilled throughout while the uncalibrated "
        "LR baseline does not. Verification metrics per test year (n = 3,004 rows in total; "
        "pooled row at the foot of each block) under the shared evaluation protocol: "
        "identical rows, identical RI label, fixed development climatology "
        f"(2000--2019 RI rate {dev_txt}), and the operational threshold taken as the "
        "inner-year F1 optimum for each model."
    )
    note = ("BSS$_{\\mathrm{dev}}$, Brier skill score against the fixed development climatology "
            f"(2000--2019 RI rate {dev_txt}), identical for every model and year; POD, "
            "probability of detection; POFD, probability of false detection (equals "
            "FARate); FARatio, false-alarm ratio; CSI, critical success index. Individual-year "
            "counts (511--757 rows) limit the precision of year-to-year contrasts; the pooled "
            "row is the primary basis for comparison. Outside the uncalibrated LR baseline, "
            "whose BSS$_{\\mathrm{dev}}$ is negative in all five years, the "
            f"only negative year-cell is the 21-predictor panel LDA in 2022 ({lda_2022}).")
    tex = table("S3", caption, r"@{}lrrrrrr@{}",
                ["Year", r"\textit{n}"] + S2_HEADERS, body, note,
                rel(EVAL_DIR) + "/{main_2024,baseline_lr,baseline_lda,"
                                "baseline_ships12_lda,baseline_ships12_lda_cohc}.json")
    return tex, n_data


# --------------------------------------------------------------------------- #
# S4 - ensemble-size sensitivity
# --------------------------------------------------------------------------- #
S3_ROWS = [("k4_2024", "k4", "K4 (Top-4)"), ("main_2024", "main", "Top-5 (main)"),
           ("k6_2024", "k6", "K6 (Top-6)"), ("all22_2024", "all22", "all-22")]
S3_HEADERS = ["BSS$_{\\mathrm{dev}}$", "ROC AUC", "PR AUC", "POD", "POFD", "FARatio", "CSI"]
S3_METRICS = ["BSS_dev", "ROC_AUC", "PR_AUC", "POD", "FARate", "FARatio", "CSI"]
DOC_S3 = {
    "k4": [0.2595, 0.898, 0.513, 0.667, 0.087, 0.621, 0.319],
    "main": [0.2590, 0.900, 0.518, 0.680, 0.095, 0.635, 0.311],
    "k6": [0.2789, 0.902, 0.527, 0.662, 0.076, 0.589, 0.339],
    "all22": [0.2438, 0.905, 0.509, 0.743, 0.128, 0.684, 0.285],
}


def build_s3() -> tuple[str, int]:
    sel = read_json(SELECTION_JSON)
    k = {key: len(sel["sensitivity"][key]["members"]) for key in DOC_S3 if key != "main"}
    k["main"] = len(sel["members"])
    body, actual = [], {}
    for fname, short, label in S3_ROWS:
        d = read_json(EVAL_DIR / f"{fname}.json")
        m = d["metrics"]["all"]
        cells = [esc(label), str(k[short]), count(m["n"])]
        for metric in S3_METRICS:
            actual[(short, metric)] = float(m[metric])
            cells.append(fmt(m[metric]))
        body.append(_body_row(cells))

    check("S3", "K values match the locked sensitivity definitions",
          k == {"k4": 4, "main": 5, "k6": 6, "all22": 22}, str(k))
    check("S3", "all ensemble sizes evaluate the same 3,004 rows",
          len({read_json(EVAL_DIR / f"{f}.json")["n_rows"] for f, _, _ in S3_ROWS}) == 1,
          "n_rows = 3004")
    expected = {(short, metric): want[i]
                for short, want in DOC_S3.items()
                for i, metric in enumerate(S3_METRICS)}
    check_map("S3", "metrics match RESULTS Sec. 4", actual, expected, 6e-4)
    span = max(actual[(s, "BSS_dev")] for s in ("k4", "main", "k6")) - \
        min(actual[(s, "BSS_dev")] for s in ("k4", "main", "k6"))
    check("S3", "caption claim: K=4-6 BSS_dev spread is below 0.03",
          span < 0.03, f"spread={span:.4f}")

    caption = (
        "K6 attains BSS$_{\\mathrm{dev}}$ 0.279, compared with 0.259 for K4 and Top-5, "
        "while all-22 attains 0.244. Full metric panel for the four ensemble sizes "
        "evaluated on the same 3,004 test rows under the locked protocol; members enter "
        "with uniform weights and each ensemble carries its own inner-year F1 threshold."
    )
    note = ("K, number of members; the K6 arm adds the sixth-ranked inner arm "
            "(ens\\_lbB\\_lpft4\\_sf) to the protocol-locked Top-5, and all-22 pools every "
            "trained arm. K4/K6 and all-22 are sensitivity analyses: the protocol-locked "
            "Top-5 remains the primary ensemble even where K6 scores higher.")
    tex = table("S4", caption, r"@{}lrrrrrrrrr@{}",
                ["Ensemble", r"\textit{K}", r"\textit{n}"] + S3_HEADERS, body, note,
                rel(EVAL_DIR) + "/{k4_2024,main_2024,k6_2024,all22_2024}.json")
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S9 - SHIPS-12 predictors
# --------------------------------------------------------------------------- #
S4_EN = {
    "VMAX": ("Maximum sustained wind (kt)", "Panel VMAX (IBTrACS USA\\_WIND, 1-min)", "EXACT"),
    "INCV": ("6-h intensity change, t$-$6~h to t0 (kt)", "Panel PER (6-h IBTrACS track difference)", "EXACT"),
    "POT": ("Potential intensity minus VMAX (kt), 0--200 km",
            "tcpyPI 1.4.1 \\citep{gilford2021tcpypi} on ERA5 soundings, 0--200 km; "
            "Bister--Emanuel algorithm \\citep{bister2002potential}", "APPROX"),
    "OHCD": ("Ocean heat content (kJ~cm$^{-2}$), 0--100 km",
             "GLORYS12V1 daily field, closed window "
             "(source\\_time $+$ 24~h $\\leq$ t0)", "BETTER"),
    "SHRD": ("850--200 hPa shear magnitude (kt), 200--800 km",
             "ERA5 u/v annular mean difference", "EXACT"),
    "T150": ("150 hPa temperature ($^{\\circ}$C), 200--800 km",
             "Log-p extrapolation from 200/300 hPa (no local 150 hPa level)", "SUBSTITUTED"),
    "T200": ("200 hPa temperature ($^{\\circ}$C), 200--800 km",
             "ERA5 t at 200 hPa", "EXACT"),
    "T250": ("250 hPa temperature ($^{\\circ}$C), 200--800 km",
             "Log-p interpolation of 200/300 hPa (244.9 hPa)", "INTERP"),
    "R000": ("1000 hPa relative humidity (\\%), 200--800 km",
             "ERA5 r at 1000 hPa", "EXACT"),
    "SDIR": ("Standard deviation of IR brightness temperature ($^{\\circ}$C), 0--200 km",
             "Paired HURSAT-B1 TBB frame (QC v1 corpus)", "NEAR-EXACT"),
    "Z850": ("850 hPa relative vorticity ($10^{-7}$~s$^{-1}$), 0--1000 km",
             "ERA5 u/v central difference, signed dlat", "EXACT"),
    "DTL": ("Great-circle distance to nearest land (km)",
            "Panel LAND\\_DIST (GSHHG coastline)", "APPROX"),
}


def build_s4() -> tuple[str, int]:
    src = BASELINE_COMMON_PY.read_text(encoding="utf-8")
    m = re.search(r"SHIPS_PREDICTORS\s*=\s*\((.*?)\)", src, re.S)
    if not m:
        raise SystemExit(f"{rel(BASELINE_COMMON_PY)}: SHIPS_PREDICTORS not found")
    locked = tuple(re.findall(r'"([A-Z0-9]+)"', m.group(1)))
    doc = md_table(SHIPS12_MD, "Predictor definitions")
    doc_names = []
    for row in doc:
        if len(row) < 5:
            continue
        name = strip_md(row[1])
        if name.endswith("COHC"):
            continue
        doc_names.append(name)
    check("S4", "documented predictor list matches src/baseline_common.py",
          tuple(doc_names) == locked,
          f"doc={doc_names} vs code={list(locked)}")
    check("S4", "table covers the twelve locked predictors",
          set(S4_EN) == set(locked), f"{len(S4_EN)} rows")

    fidelity_bad = []
    doc_fid = {strip_md(r[1]): strip_md(r[4]) for r in doc if len(r) >= 5}
    for name, (_, _, fid) in S4_EN.items():
        if fid.split("-")[0] not in doc_fid.get(name, ""):
            fidelity_bad.append(f"{name}: doc={doc_fid.get(name)!r} table={fid}")
    check("S4", "fidelity labels agree with SHIPS12_BASELINE.md Sec. 2", not fidelity_bad,
          "; ".join(fidelity_bad) or "12/12 agree")

    body = []
    for i, name in enumerate(locked, start=1):
        definition, origin, fid = S4_EN[name]
        body.append(_body_row([str(i), rf"\texttt{{{esc(name)}}}", definition, origin, fid]))

    labels = [S4_EN[n][2] for n in locked]
    check("S4", "caption claim: six exact and two near-exact predictors",
          labels.count("EXACT") == 6
          and sum(l in ("INTERP", "NEAR-EXACT") for l in labels) == 2,
          "EXACT=" + str(labels.count("EXACT")) + ", near-exact="
          + str(sum(l in ("INTERP", "NEAR-EXACT") for l in labels))
          + ", BETTER=" + str(labels.count("BETTER")))

    caption = (
        "Six of the twelve SHIPS predictors are reproduced exactly, one near-exactly "
        "and one by interpolation; the remaining substitutions are recorded below. The twelve predictors of Horinouchi et "
        "al.\\ (2025) as re-derived on our own panel, with the radius of the area average, "
        "the source used here and the fidelity label recorded in the baseline "
        "documentation."
    )
    note = ("Radii, units and the area-average operator follow WRPS1 Table 1; all fields "
            "are taken at the t0 synoptic time (0-h match within a 3-h tolerance). "
            "Fidelity labels: EXACT, direct ERA5/HURSAT/panel equivalent; NEAR-EXACT, "
            "close correspondence with a documented source difference; INTERP, log-p "
            "interpolation; APPROX, local approximation; SUBSTITUTED, layer unavailable "
            "locally and replaced; BETTER, time-varying where WRPS1 uses a climatology. "
            "The protocol-locked COHC sensitivity replaces OHCD by a fit-only "
            "(2000--2017) month-by-0.5$^{\\circ}$ GLORYS climatology; the two ocean "
            "variants are reported side by side and never enter the same model.")
    tex = table("S9", caption,
                r"@{}rlp{4.5cm}p{5.0cm}l@{}",
                ["\\#", "Predictor", "Definition (radius)", "Source in this study",
                 "Fidelity"], body, note, rel(SHIPS12_MD) + " Sec. 2 (definitions); "
                                                    + rel(BASELINE_COMMON_PY) + " (locked list)",
                tabcolsep=4)
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S2 - 22-arm inner ranking
# --------------------------------------------------------------------------- #
def build_s5() -> tuple[str, int]:
    rows = read_csv(SELECTION_CSV)
    sel = read_json(SELECTION_JSON)
    chosen = {mm["run_id"]: float(mm.get("weight") or 0.0) for mm in sel["members"]}
    thresh = {mm["run_id"]: float(mm["inner_threshold"]) for mm in sel["members"]}
    rows.sort(key=lambda r: int(r["rank"]))
    check("S5", "22 arms ranked", len(rows) == 22, f"{len(rows)} rows")
    check("S5", "five members selected with uniform weights",
          len(chosen) == 5 and len(set(round(w, 6) for w in chosen.values())) == 1,
          f"weights={sorted(set(round(w, 3) for w in chosen.values()))}")

    body, two_year, actual = [], [], {}
    for r in rows:
        arm, family = r["arm"], r["family"]
        per = json.loads(r["per_year"].replace("'", '"'))
        b18, b19 = float(per["2018"]), float(per["2019"])
        worst = float(r["two_year_worst"])
        actual[(arm, "two_year_worst")] = worst
        if abs(worst - min(b18, b19)) > 1e-9:
            two_year.append(arm)
        is_sel = arm in chosen
        body.append(_body_row([
            r["rank"], rf"\texttt{{{esc(arm)}}}", esc(family),
            fmt(thresh.get(arm, 0.225)) if is_sel else "--",
            fmt(b18), fmt(b19),
            rf"\textbf{{{fmt(worst, 4)}}}" if is_sel else fmt(worst, 4),
            rf"\textbf{{yes}}" if is_sel else "no",
        ]))
    check("S5", "ranking key is the worst of the two inner years", not two_year,
          "; ".join(two_year) or "22/22 consistent")
    top5 = {r["arm"] for r in rows if int(r["rank"]) <= 5}
    check("S5", "selected arms are exactly inner ranks 1-5", top5 == set(chosen),
          ", ".join(sorted(chosen)))
    check("S5", "rank order is monotone in the worst inner year",
          all(float(rows[i]["two_year_worst"]) >= float(rows[i + 1]["two_year_worst"])
              for i in range(len(rows) - 1)), "22/22 monotone")

    pooled_sel = float(sel["inner_pooled_bss_selected"])
    best_single = max(float(r["pooled_inner_bss"]) for r in rows)
    families = sorted({r["family"] for r in rows})
    used = sorted({r["family"] for r in rows if r["arm"] in chosen})
    check("S5", "pooled inner BSS of the selected ensemble matches RESULTS Sec. 2",
          abs(pooled_sel - 0.3324) < 6e-4, f"{pooled_sel:.4f}")
    check("S5", "best single arm pooled inner BSS matches RESULTS Sec. 2",
          abs(best_single - 0.3158) < 6e-4, f"{best_single:.4f}")
    check("S5", "caption claim: ensemble beats its best member on the inner years",
          pooled_sel > best_single, f"{pooled_sel:.4f} > {best_single:.4f}")

    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    caption = (
        "Ranking the 22 arms by their worse inner year selects five arms from "
        f"{words[len(used)]} of the {words[len(families)]} families, and the resulting "
        f"uniform-weight ensemble scores a pooled inner BSS of {pooled_sel:.3f}, above the "
        f"highest single-arm pooled inner BSS ({best_single:.3f}). Inner-year selection table: per-year inner "
        "BSS for 2018 and 2019, the worst of the two years used as the ranking key, the "
        "inner-year F1 threshold of each selected member and whether the arm entered the "
        "production ensemble."
    )
    # family gate as actually locked in selection.json
    gate = int(sel["selection_rule"]["family_gate"])
    gated = list(sel["gated_families"])
    rank_of = {r["arm"]: int(r["rank"]) for r in rows}
    family_arms = {}
    for r in rows:
        family_arms.setdefault(r["family"], []).append(r["arm"])
    champs = {fam: min(arms, key=lambda a: rank_of[a]) for fam, arms in family_arms.items()}
    gate_violations = [f"{fam}: champion {champs[fam]} at rank {rank_of[champs[fam]]} "
                       f"(gate {gate}) but a member was selected"
                       for fam in gated if rank_of[champs[fam]] > gate
                       and any(a in chosen for a in family_arms[fam])]
    check("S5", "family gate in the note matches selection.json and the ranking",
          gate == 8 and gated == ["mae_sf", "mae_mf"] and not gate_violations,
          f"gate={gate}, gated={gated}, champions="
          + ", ".join(f"{f}:{champs.get(f)}#{rank_of.get(champs.get(f), '?')}" for f in gated))

    note = ("Selection rule (protocol\\_locks/selection.json): per-year inner BSS for "
            "2018 and 2019, ranked by the worse year, tie-broken by inner mean, then "
            f"family order and arm name. Family gate ({gate}): a gated-family champion "
            "receives a protected slot only if it ranks within the overall "
            f"top {gate} by two-year-worst inner BSS; otherwise the family is not admitted, "
            "and remaining slots are filled by overall rank. Both MAE champions fell outside "
            "the gate (ranks 12 and 22), so the five highest-ranked arms form the ensemble. "
            "K~=~5, uniform weights, ensemble threshold from the inner F1 optimum. "
            "Thresholds are shown only for the five selected members. "
            "Arms are LingBot/MAE checkpoints differing in "
            "fine-tuning mode, LoRA rank, temporal pooling and input channels. "
            "For the historical run IDs ens\\_ep40ft and ens\\_ep40\\_temp5, "
            "ep40 is a run label, not an epoch-40 checkpoint identifier; the "
            "visual branch loads mae\\_vit\\_s16\\_best.pt. "
            "Short names used in the main text, figures and Tables S5--S6: "
            "LoRA-sf = ens\\_lbB\\_lora16\\_sf, LoRA-T5 = ens\\_lbB\\_lora16\\_temp5, "
            "Part4 = ens\\_lbB\\_partial4\\_sf, T2 = ens\\_lbB\\_temp2, "
            "T5 = ens\\_lbB\\_temp5\\_slim.")
    tex = "\\clearpage\n" + longtable(
        "S2", caption, r"@{}rllrrrrr@{}",
        ["Rank", "Arm", "Family", "Threshold", "BSS 2018", "BSS 2019",
         "Worst year", "Selected"], body, note,
        rel(SELECTION_CSV) + "; " + rel(SELECTION_JSON))
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S5 - 40-entity ablation panel
# --------------------------------------------------------------------------- #
S6_MEMBERS = ["ens_lbB_partial4_sf", "ens_lbB_lora16_temp5", "ens_lbB_temp2",
              "ens_lbB_lora16_sf", "ens_lbB_temp5_slim"]
SHORT = {"ens_lbB_lora16_sf": "LoRA-sf", "ens_lbB_lora16_temp5": "LoRA-T5",
         "ens_lbB_partial4_sf": "Part4", "ens_lbB_temp2": "T2",
         "ens_lbB_temp5_slim": "T5"}
S6_CONDS = ["C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf"]
S6_LABEL = {"C1": "all causal losses off", "C2": "losses and graph branch off",
            "C3s1": "prior shuffled (seed 1)", "C3s2": "prior shuffled (seed 2)",
            "C3s3": "prior shuffled (seed 3)", "C4adj": "adjacency term only",
            "C4sign": "sign term only", "C4cf": "counterfactual term only"}
DOC_S6 = {
    ("ens_lbB_lora16_sf", "C1"): 0.0065, ("ens_lbB_lora16_sf", "C2"): 0.0796,
    ("ens_lbB_lora16_sf", "C3s1"): 0.0096, ("ens_lbB_lora16_sf", "C3s2"): 0.0621,
    ("ens_lbB_lora16_sf", "C3s3"): 0.0549, ("ens_lbB_lora16_sf", "C4adj"): 0.0542,
    ("ens_lbB_lora16_sf", "C4sign"): 0.0663, ("ens_lbB_lora16_sf", "C4cf"): 0.0499,
    ("ens_lbB_partial4_sf", "C1"): -0.0933, ("ens_lbB_partial4_sf", "C4sign"): -0.0893,
    ("ens_lbB_temp2", "C3s3"): 0.0677, ("ens_lbB_temp2", "C4sign"): -0.0390,
}


def build_s6() -> tuple[str, int]:
    body, actual, sig, missing = [], {}, [], []
    for member in S6_MEMBERS:
        path = EVAL_DIR / f"M0_{member}.json"
        d = read_json(path)
        pb = d["paired_bootstrap"]
        for cond in S6_CONDS:
            key = f"abl_{cond}_{member}"
            if key not in pb:
                missing.append(key)
                continue
            v = pb[key]
            actual[(member, cond)] = float(v["delta"])
            if v["reject_holm"]:
                sig.append((member, cond))
            ci = ("$[$" + fmt(v["ci_lo"]) + ", " + fmt(v["ci_hi"]) + "$]$")
            body.append(_body_row([
                rf"\texttt{{{esc(SHORT[member])}}}", esc(cond), fmt(v["delta"]), ci,
                pfmt(v["p_holm"]), rf"\textbf{{yes}}" if v["reject_holm"] else "no"]))

    check("S6", "40 member-by-condition contrasts present", not missing,
          f"{len(body)} rows" if not missing else f"missing {missing[:4]}")
    check("S6", "ten contrasts survive Holm correction", len(sig) == 10,
          f"{len(sig)} significant: " + ", ".join(f"{m.replace('ens_lbB_', '')}/{c}"
                                                 for m, c in sig))
    members_sig = sorted({m.replace("ens_lbB_", "") for m, _ in sig})
    check("S6", "caption claim: all significant contrasts sit in three members",
          len(members_sig) == 3, ", ".join(members_sig))
    c2_flags = []
    for member in S6_MEMBERS:
        d = read_json(EVAL_DIR / f"M0_{member}.json")
        c2_flags.append(bool(d["paired_bootstrap"][f"abl_C2_{member}"]["reject_holm"]))
    check("S6", "protocol-locked C2 contrast is significant in exactly one member",
          sum(c2_flags) == 1, f"{sum(c2_flags)}/5 members")
    check_map("S6", "the twelve contrasts quoted in RESULTS Sec. 5", actual, DOC_S6, 6e-4)
    # Sec. 5 also states direction/significance summaries for two members
    lora = {c: actual[("ens_lbB_lora16_sf", c)] for c in S6_CONDS}
    slim = {c: actual[("ens_lbB_temp5_slim", c)] for c in S6_CONDS}
    lt5 = {c: actual[("ens_lbB_lora16_temp5", c)] for c in S6_CONDS}
    check("S6", "RESULTS Sec. 5: lora16_temp5 deltas span -0.026..+0.051 and never reject",
          min(lt5.values()) >= -0.027 and max(lt5.values()) <= 0.052,
          f"[{min(lt5.values()):+.3f}, {max(lt5.values()):+.3f}]")
    check("S6", "RESULTS Sec. 5: temp5_slim deltas span -0.022..+0.019",
          min(slim.values()) >= -0.023 and max(slim.values()) <= 0.020,
          f"[{min(slim.values()):+.3f}, {max(slim.values()):+.3f}]")
    check("S6", "RESULTS Sec. 5: partial4_sf C3 shuffles all negative and null",
          all(actual[("ens_lbB_partial4_sf", c)] < 0 for c in ("C3s1", "C3s2", "C3s3")),
          ", ".join(f"{c}={actual[('ens_lbB_partial4_sf', c)]:+.3f}"
                    for c in ("C3s1", "C3s2", "C3s3")))
    # Sec. 5 says the CI of these two lora16_temp5 contrasts excludes zero while Holm
    # does not reject; at 3 decimals one lower bound rounds to 0.000, which is only a
    # display artefact (recorded as a caveat, not a data mismatch).
    lb_round = []
    for cond in ("C3s2", "C4sign"):
        v = read_json(EVAL_DIR / "M0_ens_lbB_lora16_temp5.json")["paired_bootstrap"][
            f"abl_{cond}_ens_lbB_lora16_temp5"]
        if (float(v["ci_lo"]) > 0) != (round(float(v["ci_lo"]), 3) > 0):
            lb_round.append(f"S6 lora16_temp5/{cond}: ci_lo = {float(v['ci_lo']):.2e} "
                            f"prints as {fmt(v['ci_lo'])} (CI still excludes 0)")
    CAVEATS.extend(lb_round)

    caption = (
        "Ten of the 40 member-by-condition contrasts survive Holm correction and all ten "
        "fall in three of the five members, while the protocol-locked C2 contrast (graph "
        "branch removed) reaches Holm significance in LoRA-sf. Full ablation panel: "
        "delta BSS (M0 minus ablation, positive = the causal prior helps) with "
        + ABL_CI_LEVEL + r"\% storm-cluster bootstrap intervals and Holm-adjusted p-values, "
        "for five ensemble members each ablated in eight conditions on the 3,004 test rows."
    )
    note = ("Cluster bootstrap over 131 storms, 2,000 resamples, "
            + ABL_CI_LEVEL + r"\% CI; Holm correction within each member's eight-condition "
            "family at $\\alpha = 0.10$, with adjustment confined to each member's block. "
            "Conditions: C1, all causal losses removed; C2, causal losses "
            "and the graph branch removed; C3s1--C3s3, prior shuffled with each node's "
            "degree preserved in the environmental subgraph and loss weights unchanged, "
            "with three fixed "
            "seeds; C4adj, adjacency term only; C4sign, sign term only; C4cf, "
            "counterfactual term only. ``yes'' marks Holm rejection; direction of the "
            "estimate is given by the sign of delta BSS.")
    tex = longtable("S5", caption, r"@{}llrrrc@{}",
                    ["Member", "Condition", r"$\boldsymbol{\Delta}$\textbf{BSS}",
                     f"{ABL_CI_LEVEL}\\% CI", r"$\boldsymbol{p}_{\mathrm{Holm}}$",
                     "Significant"], body, note,
                    rel(EVAL_DIR) + "/M0_ens_lbB_*.json (paired_bootstrap)")
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S7 - sample-size census
# --------------------------------------------------------------------------- #
S7_EN = {
    3004: ("Official test set (main table)", "All headline numbers"),
    2892: ("WRPS1 same-sample subset", "WRPS1 delta-BSS bootstrap"),
    2385: ("RITCF reconstruction, CMA 30-kt frame", "RITCF CMA 30-kt protocol"),
    2178: ("Panel A intersection", "Consistent-comparison panel, our label standard"),
    1316: ("2020--2021 subset", "Two-year supplementary contrast"),
    1149: ("RITCF as published", "Their reported operating point (external anchor)"),
    1057: ("RITCF reconstruction, CMA 25.27-kt frame", "Independent reconstruction frame"),
    949: ("Panel B intersection", "Consistent-comparison panel, their label standard"),
}


def build_s7() -> tuple[str, int]:
    doc = md_table(RESULTS_MD, "Sample frames")
    parsed = []
    for row in doc:
        if len(row) < 3:
            continue
        n_raw = strip_md(row[1]).replace(",", "")
        try:
            parsed.append((strip_md(row[0]), int(n_raw), strip_md(row[2])))
        except ValueError:
            continue
    check("S7", "census table parsed with eight definitions", len(parsed) == 8,
          f"{len(parsed)} rows")
    check("S7", "census counts match the documented values",
          [n for _, n, _ in parsed] == list(S7_EN), f"{[n for _, n, _ in parsed]}")
    check("S7", "every census count is accounted for in the lookup table",
          set(S7_EN) == {n for _, n, _ in parsed}, "8/8 mapped to English labels")

    body = []
    for label_text, n, _ in parsed:
        label, use = S7_EN[n]
        cells = [label, count(n), use]
        if "common panel" in label_text:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        body.append(_body_row(cells))
    check("S7", "the two highlighted panels are A (2,178) and B (949)",
          [n for lbl, n, _ in parsed if "common panel" in lbl] == [2178, 949],
          "panel A and panel B intersections")

    official = next(n for lbl, n, _ in parsed if n == 3004)
    subsets = [n for _, n, _ in parsed if n < official]
    check("S7", "all reported frames are smaller than the official test set",
          all(s < official for s in subsets), f"max subset={max(subsets):,} < {official:,}")

    inhouse = [3004, 2892, 2178, 1316, 949]
    ritcf = [2385, 1149, 1057]
    counts_seen = [n for _, n, _ in parsed]
    check("S7", "caption claim: five in-house frames nested under 3,004, three RITCF frames",
          [n for n in counts_seen if n in inhouse] == inhouse
          and [n for n in counts_seen if n in ritcf] == ritcf,
          f"in-house={inhouse}, RITCF={ritcf}")

    caption = (
        "Eight sample frames are used in this study and are not interchangeable. "
        "The five in-house frames (3,004, 2,892, 2,178, 1,316 and 949 rows) "
        "are subsets of the official test table; 2,178 and 1,316 overlap, and 949 is their "
        "common subset. The two reconstructed RITCF frames (2,385 and 1,057 rows) are used "
        "for the protocol reconstructions, and the consistent-comparison panels (2,178 and "
        "949 rows) are their intersections with our frozen rows. The 1,149-row frame is the "
        "external anchor reported by \\citet{wang2025ritcf}. Sample-size census, with the "
        "use to which each row count belongs."
    )
    note = ("Counts are rows surviving frame availability and the all-finite SHIPS-12 "
            "filter, where that filter applies; the two reconstructed RITCF frames "
            "(2,385 and 1,057 rows) are intersected with our frozen test rows before any "
            "comparison (Supplementary Table S8), and 1,149 is the row count of the "
            "published operating point, which is used as an external anchor only.")
    tex = table("S7", caption, r"@{}lrr@{}", ["Sample definition", r"\textit{n}", "Use"],
                body, note, rel(RESULTS_MD) + " Sec. 5 (sample frames)")
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S8 - wind-source label control
# --------------------------------------------------------------------------- #
def build_s8() -> tuple[str, int]:
    cov: dict[str, dict] = {}
    uncovered = []
    for r in read_csv(COVERAGE_CSV):
        if not r.get("figure"):
            continue
        if r.get("ri_row_coverage"):  # coverage summary rows; the rest are uncovered-reason rows
            cov.setdefault(r["figure"], r)
        elif r.get("label"):
            uncovered.append((r["figure"], r["rows"], r["label"]))
    lock = read_json(COMPARISON_LOCK)["wind_source_decomposition"]
    waterfall = read_csv(WATERFALL_CSV)

    def rows_of(fig, stage):
        return next(r for r in waterfall if r["figure"] == fig and r["stage"] == stage)

    a_n = int(cov["A"]["rows"])
    a_cma = int(float(cov["A"]["ri_rows_total"]))
    a_usa = int(float(lock["n_RI_USA30"]))
    b_n = int(cov["B"]["rows"])
    b_cma = int(float(cov["B"]["ri_rows_total"]))
    b_usa = int(float(rows_of("B", "2_and_ships12_all_finite")["n_RI_USA30"]))
    jacc = float(lock["jaccard_RI_rows"])
    agree = float(lock["label_agreement"])

    check("S8", "panel A row counts match coverage.csv",
          a_n == 2178 and a_cma == 120 and a_usa == 209, f"n={a_n}, CMA={a_cma}, USA={a_usa}")
    check("S8", "panel B row counts match coverage.csv",
          b_n == 949 and b_cma == 66 and b_usa == 83, f"n={b_n}, CMA={b_cma}, USA={b_usa}")
    check("S8", "RI coverage of both label sets is complete in both panels",
          all(float(cov[f]["ri_row_coverage"]) == 1.0 and float(cov[f]["storm_coverage"]) == 1.0
              for f in ("A", "B")), "A: 120/120 rows, 120/120 storms; B: 66/66, 51/51")
    check("S8", "RESULTS Sec. 10: Jaccard 0.53 and row agreement 95.4\\%",
          abs(jacc - 0.53) < 5e-3 and abs(agree - 0.954) < 5e-4,
          f"Jaccard={jacc:.4f}, agreement={agree:.4f}")
    # algebra: Jaccard + counts fix the intersection, which must reproduce the agreement
    inter = jacc * (a_usa + a_cma) / (1.0 + jacc)
    back = 1.0 - ((a_usa + a_cma) - 2.0 * round(inter)) / a_n
    check("S8", "Jaccard and row agreement are mutually consistent",
          abs(back - agree) < 1e-3,
          f"implied intersection={inter:.1f} of union {a_usa + a_cma - round(inter):.0f} "
          f"-> agreement {back:.4f}")
    printed = {"jaccard": fmt(jacc), "agreement": fmt(agree),
               "rate_cma": fmt(a_cma / a_n), "rate_usa": fmt(a_usa / a_n)}
    check("S8", "printed 3-dp values match the prose in RESULTS Sec. 10",
          printed == {"jaccard": "0.530", "agreement": "0.954",
                      "rate_cma": "0.055", "rate_usa": "0.096"}, str(printed))

    body = []
    for fig, n, usa, cma in (("A", a_n, a_usa, a_cma), ("B", b_n, b_usa, b_cma)):
        body.append(_body_row([
            "A (our standard)" if fig == "A" else "B (their standard)",
            count(n), str(usa), str(cma), fmt(usa / n), fmt(cma / n),
            fmt(jacc) if fig == "A" else "--",
            fmt(agree) if fig == "A" else "--",
        ]))

    caption = (
        "The two wind sources assign the same RI label to 95.4\\% of the 2,178 panel-A "
        "rows; 114 of the 215 rows positive under either source coincide (Jaccard 0.530). "
        "Panel A compares USA-30 with CMA-30 (209 and 120 RI rows; base rates 0.096 and "
        "0.055), and panel B compares USA-30 with CMA-25.27 (83 and 66; 0.087 and 0.070). "
        "RI counts and base rates are shown for both panels; RI-set Jaccard and "
        "row-level label agreement are shown for panel A only."
    )
    note = ("Panel A, rows of our frozen test set intersecting the RITCF CMA 30-kt "
            "reconstruction and all-finite SHIPS-12 rows (2020--2024); panel B, the "
            "CMA 25.27-kt reconstruction frame intersected with our frozen predictions and "
            "all-finite SHIPS-12 rows (2020--2021). Base rates, Jaccard and agreement are "
            "fractions of the panel rows; Jaccard is computed on the RI-positive rows of "
            "panel A and agreement is the fraction of its rows whose binary RI label is "
            "identical under the two wind sources. For panel B only the per-source RI counts "
            "are reported because the two label sets are not both locked on that frame. "
            "The lower CMA base rate is part of the label shift in the cross-label controls.")
    tex = table("S8", caption, r"@{}lrrrrrrr@{}",
                ["Panel", r"\textit{n}", "RI USA", "RI CMA", "Rate USA",
                 "Rate CMA", "Jaccard", "Agreement"], body, note,
                rel(COVERAGE_CSV) + "; " + rel(COMPARISON_LOCK) +
                " (wind_source_decomposition); " + rel(RESULTS_MD) + " Sec. 4")
    return tex, len(body)


# --------------------------------------------------------------------------- #
# S6 - gate decomposition
# --------------------------------------------------------------------------- #
DOC_S9 = {
    "ens_lbB_partial4_sf": [0.629, 0.689, 0.189, 0.426, 0.039],
    "ens_lbB_temp2": [0.620, 0.299, -0.201, 0.188, 0.062],
    "ens_lbB_temp5_slim": [0.488, 0.260, -0.240, 0.141, 0.020],
    "ens_lbB_lora16_sf": [0.438, 0.230, -0.270, 0.091, -0.026],
    "ens_lbB_lora16_temp5": [0.698, 0.251, -0.249, 0.175, -0.035],
}
S9_COLS = ["gate_mean", "causal_attn_share", "causal_attn_share_minus_0p5",
           "causal_influence_gate_x_share", "gate_mean_ri_delta"]


def build_s9() -> tuple[str, int]:
    rows = [r for r in read_csv(GATE_CSV)
            if r["entity"] == "main" and r["kind"] == "ensemble"]
    check("S9", "five ensemble members in the gate table", len(rows) == 5,
          f"{len(rows)} rows")
    rows.sort(key=lambda r: -float(r["causal_attn_share"]))

    body, actual, bad = [], {}, []
    for r in rows:
        member = r["member"]
        share = float(r["causal_attn_share"])
        if abs(float(r["causal_attn_share_minus_0p5"]) - (share - 0.5)) > 1e-9:
            bad.append(member)
        for col in S9_COLS:
            actual[(member, col)] = float(r[col])
        body.append(_body_row([
            rf"\texttt{{{esc(SHORT[member])}}}", fmt(r["gate_mean"]), fmt(share),
            fmt(float(r["causal_attn_share_minus_0p5"])),
            fmt(r["causal_influence_gate_x_share"]),
            fmt(r["gate_mean_ri"]), fmt(r["gate_mean_non_ri"]),
            fmt(r["gate_mean_ri_delta"]),
        ]))
    check("S9", "share-0.5 column is the share minus the no-preference line", not bad,
          "; ".join(bad) or "5/5 consistent")
    expected = {(m, col): want[i] for m, want in DOC_S9.items()
                for i, col in enumerate(S9_COLS)}
    check_map("S9", "gate decomposition matches RESULTS Sec. 6", actual, expected, 6e-4)
    check("S9", "RESULTS Sec. 6: only one member sits above the 0.5 causal-share line",
          sum(1 for r in rows if float(r["causal_attn_share_minus_0p5"]) > 0) == 1,
          rows[0]["member"] + f" ({float(rows[0]['causal_attn_share']):.3f})")
    check("S9", "RESULTS Sec. 6: highest gate_mean is not the highest causal_share",
          max(rows, key=lambda r: float(r["gate_mean"]))["member"]
          != max(rows, key=lambda r: float(r["causal_attn_share"]))["member"],
          max(rows, key=lambda r: float(r["gate_mean"]))["member"])

    shares = [float(r["causal_attn_share"]) for r in rows]
    gates = [float(r["gate_mean"]) for r in rows]
    check("S9", "caption claim: causal share spans 0.230-0.689 across members",
          abs(min(shares) - 0.230) < 6e-4 and abs(max(shares) - 0.689) < 6e-4,
          f"[{min(shares):.3f}, {max(shares):.3f}]")

    caption = (
        "Members route different shares of their cross-attention to the causal graph "
        f"token (causal share {min(shares):.3f} to {max(shares):.3f}; gate mean "
        f"{min(gates):.3f} to {max(gates):.3f}); only one member's causal share sits above "
        "the 0.5 no-preference line. Gate decomposition of the five ensemble members "
        "over the 3,004 test rows: mean fusion gate, head-averaged causal-token attention "
        "share, its deviation from the two-token no-preference line, the resulting causal "
        "influence weight, and the RI-minus-non-RI gate contrast."
    )
    note = ("Per-row means over the 3,004 test rows. gate\\_mean is averaged over the 256 "
            "dimensions of the sigmoid gate on the fused environmental and causal attention "
            "output and over the test rows; causal\\_share is the head-averaged "
            "cross-attention share of the causal GNN token, where 0.5 is the no-preference "
            "line with two tokens; gate$\\times$share is the mean of the per-row product, "
            "used as a descriptive causal influence weight. All columns are descriptive "
            "mechanism summaries and carry no test of a mediation claim.")
    tex = table("S6", caption, r"@{}lrrrrrrr@{}",
                ["Member", "gate mean", "causal share", "share$-$0.5",
                 r"gate$\times$share", "gate RI", "gate non-RI",
                 r"$\boldsymbol{\Delta}$\textbf{gate}"],
                body, note, rel(GATE_CSV))
    return tex, len(body)


# --------------------------------------------------------------------------- #
# output validation and assembly
# --------------------------------------------------------------------------- #
def validate_tex(tex: str) -> list[str]:
    errors = []
    body_lines = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    non_ascii = sorted({ch for ch in "\n".join(body_lines) if ord(ch) > 127})
    if non_ascii:
        errors.append(f"non-ASCII characters present: {non_ascii}")
    stripped = re.sub(r"\$[^$]*\$", "", "\n".join(body_lines))
    for i, line in enumerate(stripped.splitlines(), start=1):
        if re.search(r"(?<!\\)_", line):
            errors.append(f"unescaped underscore in body line {i}: {line.strip()[:70]}")
    if tex.count("{") != tex.count("}"):
        errors.append(f"unbalanced braces: {tex.count('{')} vs {tex.count('}')}")
    for env in ("table", "longtable", "tabular", "document"):
        n_begin = len(re.findall(rf"\\begin\{{{env}\}}", tex))
        n_end = len(re.findall(rf"\\end\{{{env}\}}", tex))
        if n_begin != n_end:
            errors.append(f"{env}: {n_begin} begin vs {n_end} end")
    return errors


HEADER = """% ---------------------------------------------------------------------------
% Supplementary Tables S1-S9 (standalone LaTeX body; \\input it into your own
% manuscript source -- this repository ships no manuscript).
% Generated by scripts/05_eval/make_sm_tables.py -- do not edit by hand.
% Requires \\usepackage{booktabs} and \\usepackage{longtable} in the preamble;
% S1, S2 and S5 are longtables and may break across pages.
% Every number is read from a frozen artefact; the '% source:' line under each
% table names its input file.
% ---------------------------------------------------------------------------
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    builders = [
        ("S1", "Average treatment effects (full table)", build_s1),
        ("S2", "22-arm inner ranking", build_s5),
        ("S3", "Per-year metrics, 5 models x 2020-2024", build_s2),
        ("S4", "Ensemble-size sensitivity (K4/Top-5/K6/all-22)", build_s3),
        ("S5", "40-entity ablation panel", build_s6),
        ("S6", "Gate decomposition (5 members)", build_s9),
        ("S7", "Sample-size census (8 definitions)", build_s7),
        ("S8", "Wind-source label control", build_s8),
        ("S9", "SHIPS-12 predictor table", build_s4),
    ]

    parts, counts = [], []
    for tag, desc, builder in builders:
        tex, nrows = builder()
        parts.append(f"% ===== Supplementary Table {tag}: {desc} =====")
        parts.append(tex)
        if tag == "S4":
            parts.append(r"\clearpage")
        parts.append("")
        counts.append((tag, desc, nrows))

    doc = HEADER + "\n" + "\n".join(parts)
    errors = validate_tex(doc)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8", newline="\n")

    print("=" * 78)
    print(f"wrote {rel(out)}  ({len(doc.splitlines())} lines)")
    print("-" * 78)
    print(f"{'table':6}{'rows':>6}  content")
    for tag, desc, nrows in counts:
        print(f"{tag:6}{nrows:>6}  {desc}")
    print(f"{'total':6}{sum(n for _, _, n in counts):>6}  data rows in 9 tables")
    print("-" * 78)
    print("cross-checks (data vs docs vs internal invariants):")
    fails = 0
    for tag, name, ok, detail in CHECKS:
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {name}"
              + (f" -- {detail}" if detail else ""))
    print("-" * 78)
    print(f"checks: {len(CHECKS) - fails}/{len(CHECKS)} passed")
    if CAVEATS:
        print("display caveats (not data mismatches):")
        for c in CAVEATS:
            print("  ! " + c)
    if errors:
        print("OUTPUT VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
    else:
        print("output validation: ASCII only, underscores escaped, "
              "braces and environments balanced")
    print("=" * 78)
    return 0 if (fails == 0 and not errors) else 1


if __name__ == "__main__":
    sys.exit(main())
