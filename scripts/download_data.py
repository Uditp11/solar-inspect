"""Reproduce the four datasets under data/. Nothing here is committed; this script stands
in for committing the data.

    python scripts/download_data.py            # fetch anything missing, verify, extract
    python scripts/download_data.py --verify   # checksum what is already on disk, no network

D4 needs Kaggle credentials in %USERPROFILE%\\.kaggle\\kaggle.json (or ~/.kaggle/).

D3 is PV01 only. PV03 and PV08 are 7 GB across a 15-part spanned archive (.z01..z10)
that Python's zipfile cannot open; PV01 is the UAV-resolution subset and is what this
project uses. See docs/adr/.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw"

ZENODO = "https://zenodo.org/api/records/{rec}/files/{name}/content"

# (key, archive filename, url or None for kaggle, sha256, extracted dir name)
SETS = [
    (
        "d1",
        "d1_InfraredSolarModules.zip",
        "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip",
        "b82c706bc719b045ac4f8930570d81767a8a170d0998ca3e09283b585db05b5e",
        "d1",
    ),
    (
        "d2",
        "d2_thermal_pv_uav.zip",
        ZENODO.format(rec="16420123", name="Thermal%20PV%20Panel%20Detection%20Dataset%20for%20UAV%20Inspection.zip"),
        "8a7ed5ee3be038c4dbcab95f3523851f10e45cf666273abcab67d13a223f3474",
        "d2",
    ),
    (
        "d3_pv01",
        "PV01.zip",
        ZENODO.format(rec="5171712", name="PV01.zip"),
        "01cfb64edc427f6a73a81a401d791853cfaa92e6d830d1a49115c7aca753a93b",
        "d3_pv01",
    ),
    (
        "d4",
        "solar-power-generation-data.zip",
        None,  # Kaggle CLI
        "fecbfdd22aca74bd5e89c56e7ef5de4a06a02e00bc5125e6fd64ff3855690c32",
        "d4",
    ),
]

KAGGLE_SLUG = "anikannal/solar-power-generation-data"

# Where each archive's payload actually lands, and what it should be renamed to.
# D1 and D3 unpack to their own top-level folder; D2's folder name has spaces; D4 is bare CSVs.
UNPACK_AS = {
    "d1": "InfraredSolarModules",
    "d2": "Thermal PV Panel Detection Dataset for UAV Inspection",
    "d3_pv01": "PV01",
    "d4": None,  # extract straight into data/d4/
}

EXPECTED = {
    "d1": ("d1/images", 20000),
    "d2": ("d2/train/images", 235),
    "d3_pv01": ("d3_pv01", 3),          # three rooftop subsets
    "d4": ("d4", 4),                    # four CSVs
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  downloading -> {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def kaggle_download(dest_dir: Path) -> None:
    print(f"  kaggle datasets download -d {KAGGLE_SLUG}")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", KAGGLE_SLUG, "-p", str(dest_dir), "--force"],
        check=True,
    )


def extract(archive: Path, key: str) -> None:
    target = DATA / key
    if target.exists():
        print(f"  {key}/ already extracted")
        return
    inner = UNPACK_AS[key]
    print(f"  extracting {archive.name}")
    if inner is None:
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA)
        (DATA / inner).rename(target)
    macosx = DATA / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)


def check(key: str) -> bool:
    rel, n = EXPECTED[key]
    p = DATA / rel
    if not p.exists():
        print(f"  FAIL {key}: {rel} missing")
        return False
    got = len(list(p.iterdir()))
    ok = got >= n
    print(f"  {'ok  ' if ok else 'FAIL'} {key}: {rel} has {got} entries (expected >= {n})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="checksum what is on disk, no network")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    failures = []

    for key, fname, url, digest, _ in SETS:
        print(f"\n[{key}]")
        archive = RAW / fname

        if not archive.exists():
            if args.verify:
                print(f"  FAIL {archive.name} not on disk")
                failures.append(key)
                continue
            if url is None:
                kaggle_download(RAW)
            else:
                download(url, archive)

        got = sha256(archive)
        if got != digest:
            print(f"  FAIL sha256 mismatch\n    expected {digest}\n    got      {got}")
            # Kaggle re-zips server-side, so its digest is informational rather than a gate.
            if key != "d4":
                failures.append(key)
                continue
            print("  (D4 is re-zipped by Kaggle on each request; treating as a warning)")
        else:
            print(f"  sha256 ok {got[:16]}...")

        if not args.verify:
            extract(archive, key)
        if not check(key):
            failures.append(key)

    print("\n" + "=" * 60)
    if failures:
        print("INCOMPLETE:", ", ".join(sorted(set(failures))))
        return 1
    print("all four datasets present and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
