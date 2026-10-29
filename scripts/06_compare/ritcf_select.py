# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""RITCF protocol-arm ensemble selection on validation years 2018/2019.

Same rules as the main experiment (docs/ENSEMBLE_SELECTION_LITERATURE.md +
ensemble_reselect.py), registered 2026-09-19 (SERVER_OPS):
  * per-arm robust score = worst-year minimax: min(BSS_2018, BSS_2019),
    BSS vs each year's own climatology (train.py convention)
  * ranking by robust score; family-stratified Top-K with quality gate 8
    (families: lingbot_sf/lingbot_mf/mae_sf/mae_mf; mf = _temp<N>)
  * combination scan: all subsets K=2..5, uniform average of member probs,
    subset robust score = min over the two years
  * K=5 primary, K=4/6 sensitivity, uniform weights
  * selection uses ONLY validation-year predictions (train-window never
    needed; test 2020-21 evaluated once later, not here)

Run from the release root:
    python scripts_tmp/ritcf_select.py
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np

P2 = Path(__file__).resolve().parents[2]
PREDS = P2 / "data/preds"
YEARS = [2018, 2019]
FAMILIES = ["lingbot_sf", "lingbot_mf", "mae_sf", "mae_mf"]
GATE = 8


def bss_score(p, y):
    bs = float(np.mean((p - y) ** 2))
    bs_clim = float(np.mean((y.mean() - y) ** 2))
    return 1.0 - bs / max(bs_clim, 1e-9)


def arm_family(arm):
    is_lingbot = arm.startswith(("rit_lingbot", "rit_lb"))
    is_mae = arm.startswith(("rit_v1", "rit_ep40"))
    mf = re.search(r"_temp\d", arm) is not None
    if is_lingbot:
        return "lingbot_mf" if mf else "lingbot_sf"
    if is_mae:
        return "mae_mf" if mf else "mae_sf"
    return "other"


# ---- load per-arm per-year held predictions ----
arms = {}
for f in sorted(PREDS.glob("rit_*_fold*.npz")):
    m = re.match(r"(rit_.+)_fold(\d{4})\.npz", f.name)
    arm, year = m.group(1), int(m.group(2))
    if year not in YEARS:
        continue
    d = np.load(f)
    arms.setdefault(arm, {})[year] = (d["p_held"].astype(float), d["y_held"].astype(float))
arms = {a: v for a, v in arms.items() if set(v) == set(YEARS)}
print(f"arms with both validation folds: {len(arms)}")

# ---- per-arm scores ----
arm_bss = {a: {y: bss_score(*v[y]) for y in YEARS} for a, v in arms.items()}
robust = {a: min(b.values()) for a, b in arm_bss.items()}
ranking = sorted(arms, key=lambda a: -robust[a])
fams = {a: arm_family(a) for a in arms}

print("\nper-arm BSS (2018 / 2019 / robust-min / family):")
for a in ranking:
    print(f"  {a:26s} {arm_bss[a][2018]:.4f} {arm_bss[a][2019]:.4f} "
          f"{robust[a]:.4f}  {fams[a]}")

# ---- family-stratified Top-K (gate 8) ----
def family_select(ranking, k, gate):
    champs, seen, gated = [], set(), []
    for i, a in enumerate(ranking, 1):
        f = fams[a]
        if f in FAMILIES and f not in seen:
            seen.add(f)
            if i <= gate:
                champs.append(a)
            else:
                gated.append(f)
    sel = champs[:k]
    others = [a for a in ranking if fams[a] == "other"]
    if others and len(sel) < k:
        oc = others[0]
        if ranking.index(oc) + 1 <= gate:
            sel.append(oc)
    for a in ranking:
        if len(sel) >= k:
            break
        if a not in sel:
            sel.append(a)
    return sel[:k], gated

print("\nfamily-stratified selection:")
for k in (4, 5, 6):
    sel, gated = family_select(ranking, k, GATE)
    pb = {y: bss_score(np.mean([arms[a][y][0] for a in sel], axis=0), arms[sel[0]][y][1]) for y in YEARS}
    print(f"  K={k}: {sel}")
    print(f"       pooled-year BSS: 2018={pb[2018]:.4f} 2019={pb[2019]:.4f} "
          f"robust={min(pb.values()):.4f} (gated out: {gated})")

# ---- combination scan K=2..5 (uniform) ----
print("\ncombination scan (uniform, robust=min over years), top 5 per K:")
best = {}
for k in (2, 3, 4, 5):
    scored = []
    for combo in itertools.combinations(ranking, k):
        pb = {}
        for y in YEARS:
            y_ref = arms[combo[0]][y][1]
            p_ens = np.mean([arms[a][y][0] for a in combo], axis=0)
            pb[y] = bss_score(p_ens, y_ref)
        scored.append((min(pb.values()), pb, combo))
    scored.sort(key=lambda s: -s[0])
    best[k] = scored[0]
    for s, pb, combo in scored[:5]:
        print(f"  K={k} robust={s:.4f} (18={pb[2018]:.4f} 19={pb[2019]:.4f}) {list(combo)}")

print("\nbest-by-scan per K:")
for k, (s, pb, combo) in best.items():
    print(f"  K={k} robust={s:.4f}: {list(combo)}")
