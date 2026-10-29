"""Pytest entry points for the release self-checks (CPU only, no downloads).

Run from the release root:

    python -m pytest tests/ -q

Skips (rather than fails) when the data record has not been extracted yet, so a
code-only checkout still passes.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HAS_DATA = (DATA / "test" / "holdout_masked_v1" / "main.npz").is_file()
POSIX_ROOTS = ("root", "home", "Users", "mnt", "opt", "var", "workspace", "autodl-tmp")
POSIX_ROOTS_BYTES = tuple(root.encode("ascii") for root in POSIX_ROOTS)
ABS_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?=[A-Za-z0-9_ .\\-]{3,}(?:[\\/]|$))|/(?:" + "|".join(POSIX_ROOTS) + r")[\\/])")
ABS_RE_BYTES = re.compile(rb"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?=[A-Za-z0-9_ .\\-]{3,}(?:[\\/]|$))|/(?:" + b"|".join(POSIX_ROOTS_BYTES) + rb")[\\/])")


def has_build_path_bytes(blob):
    for match in re.finditer(rb"[\x20-\x7E]{4,}", blob):
        if ABS_RE_BYTES.search(match.group()):
            return True
    for match in re.finditer(rb"(?:[\x20-\x7E]\x00){4,}", blob):
        if ABS_RE.search(match.group().decode("utf-16le")):
            return True
    return False


def test_layout_is_complete():
    """Every directory promised by the release structure exists."""
    for rel in ("src/wpac_causal", "src/wpac_warning", "configs/phase1",
                "configs/phase2", "scripts/01_data", "scripts/02_causal",
                "scripts/03_assets", "scripts/04_train", "scripts/05_eval",
                "scripts/06_compare", "scripts/07_figures", "scripts/ritcf_recon",
                "protocol_locks", "docs", "figures", "requirements", "tests"):
        assert (ROOT / rel).is_dir(), rel
    for rel in ("README.md", "LICENSE-CODE", "LICENSE-DATA", "CITATION.cff",
                "figures/fig1_causal_discovery.png", "figures/fig2_architecture.pdf",
                "data/ritcf_recon/ACQUIRE.md"):
        assert (ROOT / rel).is_file(), rel
    assert not (DATA / "runs/holdout_masked_v1/runner_logs").exists()
    assert not (DATA / "runs_holdout/ablations/runner_logs").exists()
    # the paper is deliberately not part of the release
    assert not (ROOT / "paper").exists()


def test_no_absolute_build_paths_in_code():
    """The source tree must not name the build machine."""
    for p in list((ROOT / "src").rglob("*.py")) + \
            list((ROOT / "scripts").rglob("*.py")) + \
            list((ROOT / "configs").rglob("*.yaml")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert not ABS_RE.search(txt), f"{p.relative_to(ROOT)} contains an absolute build path"


def test_configs_parse():
    yaml = pytest.importorskip("yaml")
    for p in sorted((ROOT / "configs").rglob("*.yaml")):
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert isinstance(doc, dict) and doc, p


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_protocol_locks_verify():
    res = subprocess.run([sys.executable, str(ROOT / "tests" / "verify_protocol_locks.py")],
                         capture_output=True, text=True, cwd=str(ROOT))
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ALL CHECKS PASSED" in res.stdout


@pytest.mark.skipif(not (DATA / "ritcf_recon" / "dataset" / "samples.parquet").is_file(),
                    reason="RITCF data record not extracted")
def test_ritcf_report_without_large_caches(tmp_path, monkeypatch, capsys):
    dataset = DATA / "ritcf_recon" / "dataset"
    for name in ("samples.parquet", "manifest.json"):
        (tmp_path / name).write_bytes((dataset / name).read_bytes())
    from scripts.ritcf_recon import build_dataset
    monkeypatch.setattr(build_dataset.C, "dataset_dir", lambda _: tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--report-only"])
    assert build_dataset.main() == 0
    output = capsys.readouterr().out
    assert "array caches unavailable; using manifest shapes:" in output
    assert "pairs:" in output


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_headline_bss_recomputed():
    """The abstract's BSS_dev 0.259 must come back out of the frozen npz."""
    np = pytest.importorskip("numpy")
    lock = json.loads((ROOT / "protocol_locks" / "final.json").read_text(encoding="utf-8"))
    z = np.load(DATA / "test" / "holdout_masked_v1" / "main.npz", allow_pickle=True)
    p, y, thr = z["p"].astype(float), z["y"].astype(int), float(z["threshold"])
    assert len(p) == lock["evaluation"]["n_test_rows"] == 3004
    assert abs(thr - lock["threshold"]) < 1e-9
    dev_rate = float(lock["evaluation"]["dev_rate"])
    bss_dev = 1.0 - float(np.mean((p - y) ** 2)) / float(np.mean((dev_rate - y) ** 2))
    assert abs(bss_dev - 0.258973) < 1e-5, bss_dev


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_frozen_predictions_are_bound_to_the_lock():
    """The lock tolerates exactly its own digest and its recorded predecessor."""
    sys.path.insert(0, str(ROOT / "src"))
    np = pytest.importorskip("numpy")
    from wpac_warning.protocol import accepted_lock_sha256

    lock = json.loads((ROOT / "protocol_locks" / "final.json").read_text(encoding="utf-8"))
    accepted = accepted_lock_sha256(lock)
    assert lock["lock_sha256"] in accepted
    assert lock["release_reissue"]["superseded_lock_sha256"] in accepted
    assert len(accepted) == 2

    expected_members = {"main": 5, "k4": 4, "k6": 6, "all22": 22,
                        "M0_ens_lbB_partial4_sf": 1, "baseline_lr": 1}
    for name, n_members in expected_members.items():
        z = np.load(DATA / "test" / "holdout_masked_v1" / f"{name}.npz",
                    allow_pickle=True)
        assert str(z["lock_sha256"]) in accepted, name
        assert z["p_members"].shape == (3004, n_members), name


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_panel_and_manifests_agree():
    pd = pytest.importorskip("pandas")
    panel = pd.read_parquet(DATA / "holdout_masked_v1" / "panel_fit0017.parquet")
    assert panel.shape[0] == 17655
    assert set(("SID", "ISO_TIME", "row_uid" if "row_uid" in panel else "DV24")) <= set(panel.columns)
    split = pd.read_csv(DATA / "phase1" / "processed_masked" / "holdout_fit0017"
                        / "panel" / "split_manifest.csv")
    assert set(split["split"]) == {"fit", "inner", "test"}
    assert split.shape[0] == 720

    frames = pd.read_parquet(DATA / "holdout_masked_v1" / "frames_index.parquet")
    assert list(frames.columns) == ["SID", "ISO_TIME", "frame_path", "frame_time", "dt_h"]
    assert len(frames) == 17441
    assert (frames["dt_h"] == 0).all()

    pair_file = DATA / "ritcf_recon" / "dataset" / "pairs_val.parquet"
    if pair_file.is_file():
        pairs = pd.read_parquet(pair_file)
        manifest = json.loads((pair_file.parent / "manifest.json").read_text(encoding="utf-8"))
        assert len(pairs) == 3247
        assert manifest["pairs_summary"]["pairs"]["val"]["n_pairs"] == len(pairs)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str, allow_nan=False)
                          .encode("utf-8")).hexdigest()


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_selection_lock_is_self_consistent_and_chained():
    """The data-record selection lock must be a lock, not a consumer copy."""
    root = json.loads((ROOT / "protocol_locks" / "selection.json")
                      .read_text(encoding="utf-8"))
    data = json.loads((DATA / "protocol_locks" / "selection.json")
                      .read_text(encoding="utf-8"))
    fin = json.loads((ROOT / "protocol_locks" / "final.json")
                     .read_text(encoding="utf-8"))

    assert _sha({k: v for k, v in root.items() if k != "lock_sha256"}) \
        == root["lock_sha256"]
    assert data == root, "data-record copy differs from the root copy"
    assert fin["selection_sha256"] == root["lock_sha256"], "chain broken"
    assert "../wpac_causal_phase1" not in json.dumps(root)


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_asset_manifest_is_populated_and_verifies():
    man = json.loads((ROOT / "protocol_locks" / "asset_sha256_manifest.json")
                     .read_text(encoding="utf-8"))
    entries, normalised = man["entries"], man["normalised"]
    assert len(entries) >= 100, "asset manifest is (nearly) empty"
    assert man["n_mismatched"] == 0
    for key, rec in entries.items():
        rel = key.split("::", 1)[1]
        f = ROOT / rel
        assert f.is_file(), rel
        assert hashlib.sha256(f.read_bytes()).hexdigest() == rec["sha256_actual"], key
    for key, rec in normalised.items():
        rel = key.split("::", 1)[1]
        f = ROOT / rel
        assert f.is_file(), rel
        assert hashlib.sha256(f.read_bytes()).hexdigest() \
            == rec["sha256_of_shipped_copy"], key


@pytest.mark.skipif(not HAS_DATA, reason="data record not extracted")
def test_binary_sidecars_carry_no_build_path():
    """parquet metadata / pickle headers must not leak the build machine."""
    dirty = []
    for ext in ("*.parquet", "*.npz", "*.pkl", "*.npy"):
        for p in DATA.rglob(ext):
            blob = p.read_bytes()
            if has_build_path_bytes(blob):
                dirty.append(str(p.relative_to(ROOT)))
    assert not dirty, dirty[:5]
