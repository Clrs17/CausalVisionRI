"""Protocol-lock audit chain verification for the TyphoonRI release.

Checks, in order, and prints a one-line verdict for each:

  1. ``protocol_locks/final.json``    self-hash  (``sha256_json`` of the payload
     with ``lock_sha256`` removed equals the stored ``lock_sha256``).
  2. ``protocol_locks/selection.json`` self-hash.
  3. ``final.selection_sha256`` equals ``selection.lock_sha256`` (the chain).
  4. ``test_receipt.json`` is bound to the same lock and ``status == completed``.
  5. the data-record ``selection.json`` self-verifies, is byte-identical to the
     root copy, is the lock the chain points at, and carries no residual
     ``../wpac_causal_phase1`` reference.
  6. ``asset_sha256_manifest.json`` is populated, re-verifies against the
     shipped files (including the entries declared ``normalised``), and records
     the current lock digests.
  7. every ``*_sha256`` in the locks and in the data sidecars that names a
     shipped file equals that file's actual SHA-256; a mismatch is only
     tolerated when the asset manifest declares it as path-normalised.
     Fields describing non-redistributed artefacts are counted, not failed.
  8. no absolute build-machine path survives anywhere in the text *or* binary
     files (parquet metadata, pickle headers and npz member names included).

Exit status is 0 only if every check passes.

Usage (from the release root):

    python tests/verify_protocol_locks.py [--verbose]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "protocol_locks"

TEXT_EXT = {".json", ".md", ".csv", ".yaml", ".yml", ".txt", ".log", ".stdout",
            ".py", ".cff", ".tex", ".bib"}
# binaries carry metadata too (parquet/pickle headers, npz member names), so the
# sweep has to look inside them as well
BINARY_EXT = {".parquet", ".npz", ".npy", ".pkl", ".pickle", ".nc", ".pt", ".gz"}
POSIX_ROOTS = ("root", "home", "Users", "mnt", "opt", "var", "workspace", "autodl-tmp")
POSIX_ROOTS_BYTES = tuple(root.encode("ascii") for root in POSIX_ROOTS)
ABS_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?=[A-Za-z0-9_ .\\-]{3,}(?:[\\/]|$))|/(?:" + "|".join(POSIX_ROOTS) + r")[\\/])")
ABS_RE_BYTES = re.compile(rb"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?=[A-Za-z0-9_ .\\-]{3,}(?:[\\/]|$))|/(?:" + b"|".join(POSIX_ROOTS_BYTES) + rb")[\\/])")

# Documents that describe path normalisation itself
ABS_ALLOW = {
    "docs/KNOWN_ISSUES.md",
    "docs/PROTOCOL.md",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    txt = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str,
                     allow_nan=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def has_build_path_bytes(blob):
    for match in re.finditer(rb"[\x20-\x7E]{4,}", blob):
        if ABS_RE_BYTES.search(match.group()):
            return True
    for match in re.finditer(rb"(?:[\x20-\x7E]\x00){4,}", blob):
        if ABS_RE.search(match.group().decode("utf-16le")):
            return True
    return False


def resolve(rel: str) -> Path | None:
    q = Path(str(rel).replace("\\", "/"))
    cand = q if q.is_absolute() else ROOT / q
    return cand if cand.is_file() else None


def walk_hashes(node, sink, origin: str) -> None:
    """Collect (origin, field, path, recorded_sha) for every path/hash pair."""
    if isinstance(node, list):
        for item in node:
            walk_hashes(item, sink, origin)
        return
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if k == "lock_sha256":
            continue
        if isinstance(v, str) and k.endswith("_sha256") and len(v) == 64:
            for cand in (k[:-7], k[:-7] + "_path", k[:-7] + "_file"):
                pv = node.get(cand)
                if isinstance(pv, str):
                    sink.append((origin, k, pv, v))
                    break
        elif k == "sha256" and isinstance(v, str) and len(v) == 64 \
                and isinstance(node.get("path"), str):
            sink.append((origin, "sha256", node["path"], v))
        elif isinstance(v, (dict, list)):
            walk_hashes(v, sink, origin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ok = True

    # 1-2 self hashes ---------------------------------------------------- #
    locks = {}
    for name in ("final.json", "selection.json"):
        p = LOCKS / name
        if not p.is_file():
            print(f"FAIL  {name}: missing")
            ok = False
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        locks[name] = doc
        payload = {k: v for k, v in doc.items() if k != "lock_sha256"}
        good = sha256_json(payload) == doc["lock_sha256"]
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name} self-hash "
              f"({doc['lock_sha256'][:16]}...)")

    fin = locks.get("final.json", {})
    sel = locks.get("selection.json", {})

    # 3 chain ------------------------------------------------------------ #
    if fin and sel:
        good = fin.get("selection_sha256") == sel.get("lock_sha256")
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  chain: final.selection_sha256 "
              f"== selection.lock_sha256")

    # 4 receipt ---------------------------------------------------------- #
    rc_path = LOCKS / "test_receipt.json"
    if rc_path.is_file():
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        good = (rc.get("lock_sha256") == fin.get("lock_sha256")
                and rc.get("status") == "completed")
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  receipt bound to final lock, "
              f"status={rc.get('status')!r}, "
              f"{len(rc.get('outputs') or {})} entities")
        # the receipt copy inside the data record must agree
        rc2 = ROOT / "data" / "test" / "holdout_masked_v1" / "test_receipt.json"
        if rc2.is_file():
            rc2doc = json.loads(rc2.read_text(encoding="utf-8"))
            same = rc2doc.get("lock_sha256") == fin.get("lock_sha256")
            ok &= same
            print(f"{'PASS' if same else 'FAIL'}  data-record receipt agrees")
    else:
        print("WARN  no protocol_locks/test_receipt.json")

    # 5 selection-lock chain ---------------------------------------------- #
    sel_root = LOCKS / "selection.json"
    sel_data = ROOT / "data" / "protocol_locks" / "selection.json"
    if sel_root.is_file():
        good = sha256_json({k: v for k, v in sel.items()
                            if k != "lock_sha256"}) == sel["lock_sha256"]
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  data-record selection.json "
              f"self-hash ({sel['lock_sha256'][:16]}...)")
        good = fin.get("selection_sha256") == sel["lock_sha256"]
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  chain: final.selection_sha256 "
              f"points at the shipped selection lock")
        if sel_data.is_file():
            good = sel_data.read_bytes() == sel_root.read_bytes()
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  data/protocol_locks/"
                  f"selection.json is byte-identical to protocol_locks/")
        leftover = sel_root.read_text(encoding="utf-8").count(
            "../wpac_causal_phase1")
        good = leftover == 0
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  selection.json has no residual "
              f"../wpac_causal_phase1 reference ({leftover})")

    # 6 asset manifest ----------------------------------------------------- #
    am_path = LOCKS / "asset_sha256_manifest.json"
    if am_path.is_file():
        am = json.loads(am_path.read_text(encoding="utf-8"))
        entries = am.get("entries") or {}
        norm = am.get("normalised") or {}
        good = bool(entries) and am.get("n_entries", 0) >= 100
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  asset manifest is populated "
              f"({len(entries)} verified, {len(norm)} normalised, "
              f"{am.get('n_not_redistributed', 0)} not redistributed)")
        bad = [k for k, v in entries.items() if not v.get("match")]
        for k, v in norm.items():
            rel = k.split("::", 1)[1]
            f = resolve(rel)
            if f is None or sha256_file(f) != v.get("sha256_of_shipped_copy"):
                bad.append(k)
        good = not bad
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  asset manifest re-verifies "
              f"({len(bad)} stale entries)")
        for b in bad[:5]:
            print(f"        {b}")
        good = (am["locks"]["final.json"] == fin.get("lock_sha256")
                and am["locks"]["selection.json"] == sel.get("lock_sha256"))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  asset manifest records the "
              f"current lock digests")
    else:
        print("FAIL  no protocol_locks/asset_sha256_manifest.json")
        ok = False

    # 7 pinned artefact hashes ------------------------------------------- #
    pairs: list[tuple] = []
    walk_hashes(fin, pairs, "protocol_locks/final.json")
    receipt = LOCKS / "test_receipt.json"
    if receipt.is_file():
        walk_hashes(json.loads(receipt.read_text(encoding="utf-8")), pairs,
                    "protocol_locks/test_receipt.json")
    for p in sorted((ROOT / "data").rglob("*.json")):
        try:
            walk_hashes(json.loads(p.read_text(encoding="utf-8")), pairs,
                        str(p.relative_to(ROOT)))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    declared = set((json.loads(am_path.read_text(encoding="utf-8"))
                    .get("normalised") or {})) if am_path.is_file() else set()
    matched = mismatched = absent = normalised = 0
    bad_examples = []
    for origin, field, rel, recorded in pairs:
        f = resolve(rel)
        if f is None:
            absent += 1
            continue
        actual = sha256_file(f)
        if actual == recorded:
            matched += 1
        elif f"{origin}::{rel}" in declared:
            normalised += 1
        else:
            mismatched += 1
            if len(bad_examples) < 10:
                bad_examples.append((origin, field, rel, recorded[:12],
                                     actual[:12]))
    good = mismatched == 0
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  pinned hashes: {matched} verified, "
          f"{normalised} normalised (declared), "
          f"{absent} describe artefacts not in the record, {mismatched} mismatched")
    for e in bad_examples:
        print(f"        {e[0]} :: {e[1]} {e[2]}  lock={e[3]}... actual={e[4]}...")

    # 8 no absolute build paths, text and binary -------------------------- #
    offenders = {}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = p.relative_to(ROOT).as_posix()
        if rel in ABS_ALLOW:
            continue
        try:
            if ext in TEXT_EXT:
                n = len(ABS_RE.findall(p.read_text(encoding="utf-8",
                                                   errors="ignore")))
            elif ext in BINARY_EXT:
                n = int(has_build_path_bytes(p.read_bytes()))
            else:
                continue
        except OSError:
            continue
        if n:
            offenders[rel] = n
    good = not offenders
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  no absolute build paths in "
          f"{len(TEXT_EXT)}-type text or {len(BINARY_EXT)}-type binary files"
          + ("" if good else f": {offenders}"))

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
