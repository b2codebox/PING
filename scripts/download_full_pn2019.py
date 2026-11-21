"""Resumable downloader & ingester for the full PhysioNet/CinC Challenge 2019 dataset.

PhysioNet serves Challenge 2019 v1.0.0 as 40,336 individual .psv files across
training_setA (20,336) and training_setB (20,000) without a bulk .zip archive.
Run this script on a machine with ~20 minutes and ~500 MB free disk to build
the full 40,336-stay Parquet cache in ``data/processed_pn2019/``.

Usage:
    python scripts/download_full_pn2019.py --limit 0   # 0 = all 40,336 stays
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from src.ingest.loader import write_cache
from src.ingest.physionet2019 import build_physionet2019_cohort

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"


def _list_psv(subset: str) -> list[str]:
    r = requests.get(f"{BASE}/{subset}/", timeout=30)
    r.raise_for_status()
    return sorted(set(re.findall(r"p\d+\.psv", r.text)))


def _fetch_one(args: tuple[str, str, Path]) -> bool:
    subset, fname, out_dir = args
    dest = out_dir / fname
    if dest.exists() and dest.stat().st_size > 50:
        return True
    try:
        r = requests.get(f"{BASE}/{subset}/{fname}", timeout=15)
        if r.status_code == 200 and r.text.startswith("HR|"):
            dest.write_text(r.text)
            return True
    except requests.RequestException:
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max stays per set (0 = all)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/pn2019_full"))
    args = ap.parse_args()

    for subset, tag in (("training_setA", "A"), ("training_setB", "B")):
        out = args.tmp_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        files = _list_psv(subset)
        if args.limit > 0:
            files = files[: args.limit]
        print(f"Fetching {len(files)} files from {subset}...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            ok = sum(ex.map(_fetch_one, [(subset, f, out) for f in files]))
        print(f"Downloaded {ok}/{len(files)} from {subset}")

    cohort = build_physionet2019_cohort(args.tmp_dir)
    root = Path(__file__).resolve().parents[1]
    write_cache(cohort, root / "data" / "processed_pn2019")
    print(f"Saved {len(cohort.cohort)} stays to {root / 'data' / 'processed_pn2019'}")


if __name__ == "__main__":
    main()
