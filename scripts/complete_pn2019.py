"""Top up the PhysioNet 2019 cache with the stays that are not yet ingested.

Unlike ``download_full_pn2019.py`` (which re-downloads all 40,336 files and
rebuilds the cache from scratch), this script:

1. lists the .psv files actually published for each subset,
2. skips any file whose deterministic pseudonym already exists in
   ``data/processed_pn2019/cohort.parquet``,
3. downloads only the remainder,
4. ingests just those new files and merges them into the existing cache.

Usage:
    python scripts/complete_pn2019.py --dry-run      # show the gap, fetch nothing
    python scripts/complete_pn2019.py --workers 32
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.loader import Cohort, Pseudonymizer, load_cache, write_cache  # noqa: E402
from src.ingest.physionet2019 import build_physionet2019_cohort  # noqa: E402

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
CACHE_DIR = ROOT / "data" / "processed_pn2019"


def list_published(sess: requests.Session, subset: str) -> list[str]:
    """Return the .psv filenames actually listed for a subset."""
    r = sess.get(f"{BASE}/{subset}/", timeout=60)
    r.raise_for_status()
    return sorted(set(re.findall(r"p\d+\.psv", r.text)))


def already_have() -> set[str]:
    path = CACHE_DIR / "cohort.parquet"
    if not path.exists():
        return set()
    return set(load_cache(CACHE_DIR).cohort["pseudonym"])


def download_missing(
    subset: str, tag: str, files: list[str], out_dir: Path, workers: int
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers
    )
    sess.mount("https://", adapter)

    done = {"ok": 0, "fail": 0}

    def _get(fname: str) -> bool:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 50:
            return True
        for attempt in range(3):
            try:
                # PhysioNet 302-redirects to a file server; allow_redirects=True
                # is requests' default and is required here.
                r = sess.get(f"{BASE}/{subset}/{fname}", timeout=30)
                if r.status_code == 200 and r.text.startswith("HR|"):
                    dest.write_text(r.text)
                    return True
            except requests.RequestException:
                continue
        return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_get, files))

    done["ok"] = sum(results)
    done["fail"] = len(results) - done["ok"]
    return done["ok"], done["fail"]


def merge_cache(c_new: Cohort) -> Cohort:
    """Append only the new stays to the existing cache."""
    if not (CACHE_DIR / "cohort.parquet").exists():
        return c_new

    c_old = load_cache(CACHE_DIR)
    new_ids = set(c_new.cohort["pseudonym"]) - set(c_old.cohort["pseudonym"])
    if not new_ids:
        return c_old

    def _keep(df: pd.DataFrame) -> pd.DataFrame:
        if not len(df) or "pseudonym" not in df.columns:
            return df
        return df[df["pseudonym"].isin(new_ids)]

    return Cohort(
        cohort=pd.concat(
            [c_old.cohort, _keep(c_new.cohort)], ignore_index=True
        ),
        labs=pd.concat([c_old.labs, _keep(c_new.labs)], ignore_index=True),
        vitals=pd.concat([c_old.vitals, _keep(c_new.vitals)], ignore_index=True),
        vasopressors=c_old.vasopressors,
        urine=c_old.urine,
        devices=c_old.devices,
        micro=c_old.micro,
        antimicrobials=c_old.antimicrobials,
        comorbidities=c_old.comorbidities,
        issues=c_old.issues,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/pn2019_topup"))
    ap.add_argument("--dry-run", action="store_true", help="Report the gap only")
    ap.add_argument("--limit", type=int, default=0, help="Cap files per subset")
    args = ap.parse_args()

    have = already_have()
    print(f"cache currently holds {len(have)} stays")

    reserve = requests.Session()
    pseudo = Pseudonymizer()
    plan: list[tuple[str, str, list[str]]] = []
    total_missing = 0

    for subset, tag in (("training_setA", "A"), ("training_setB", "B")):
        published = list_published(reserve, subset)
        missing = [
            f
            for f in published
            if pseudo.stay(f"pn19:{tag}:{f[:-4]}") not in have
        ]
        if args.limit:
            missing = missing[: args.limit]
        total_missing += len(missing)
        print(f"{subset}: {len(published)} published, {len(missing)} missing")
        plan.append((subset, tag, missing))

    if args.dry_run:
        print(f"dry run: would download {total_missing} files")
        return
    if not total_missing:
        print("cache is already complete")
        return

    # Keep the temp dir across runs: already-downloaded .psv files are skipped,
    # so an interrupted run resumes where it stopped instead of starting over.
    if not args.tmp_dir.exists():
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

    fetched = failed = 0
    for subset, tag, missing in plan:
        if not missing:
            continue
        print(f"downloading {len(missing)} files from {subset} "
              f"with {args.workers} workers ...", flush=True)
        ok, bad = download_missing(
            subset, tag, missing, args.tmp_dir / tag, args.workers
        )
        fetched += ok
        failed += bad
        print(f"  {subset}: {ok} ok, {bad} failed", flush=True)

    if not fetched:
        print("nothing downloaded; cache unchanged")
        return

    print("ingesting new files ...", flush=True)
    c_new = build_physionet2019_cohort(args.tmp_dir)
    merged = merge_cache(c_new)
    write_cache(merged, CACHE_DIR)

    n_pos = int(merged.cohort["has_sepsis_label"].sum())
    print(
        f"DONE: fetched {fetched} (failed {failed}) | "
        f"cache now {len(merged.cohort)} stays, {n_pos} sepsis-positive"
    )

    shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
