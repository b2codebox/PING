"""Fast HTTP keep-alive batch fetcher for PhysioNet/CinC Challenge 2019.

Appends a slice of training_setA or training_setB directly into
``data/processed_pn2019/`` without leaving raw .psv files in the workspace.
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from src.ingest.loader import Cohort, load_cache, write_cache
from src.ingest.physionet2019 import build_physionet2019_cohort

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
ROOT = Path(__file__).resolve().parents[1]


def fetch_batch(subset: str, start: int, end: int, workers: int = 32) -> int:
    tag = "A" if subset == "training_setA" else "B"
    tmp_dir = Path(f"/tmp/pn2019_batch_{tag}_{start}_{end}")
    out = tmp_dir / tag
    out.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    sess.mount("https://", adapter)

    r = sess.get(f"{BASE}/{subset}/", timeout=30)
    r.raise_for_status()
    all_files = sorted(set(re.findall(r"p\d+\.psv", r.text)))
    batch = all_files[start - 1 : end]

    def _get(fname: str) -> bool:
        try:
            resp = sess.get(f"{BASE}/{subset}/{fname}", timeout=15)
            if resp.status_code == 200 and resp.text.startswith("HR|"):
                (out / fname).write_text(resp.text)
                return True
        except requests.RequestException:
            return False
        return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = sum(ex.map(_get, batch))

    c_new = build_physionet2019_cohort(tmp_dir)
    cache_dir = ROOT / "data" / "processed_pn2019"
    if (cache_dir / "cohort.parquet").exists():
        c_old = load_cache(cache_dir)
        existing_ids = set(c_old.cohort["pseudonym"])
        new_mask = ~c_new.cohort["pseudonym"].isin(existing_ids)
        new_ids = set(c_new.cohort.loc[new_mask, "pseudonym"])
        merged = Cohort(
            cohort=pd.concat([c_old.cohort, c_new.cohort[new_mask]], ignore_index=True),
            labs=pd.concat([c_old.labs, c_new.labs[c_new.labs["pseudonym"].isin(new_ids)]], ignore_index=True),
            vitals=pd.concat([c_old.vitals, c_new.vitals[c_new.vitals["pseudonym"].isin(new_ids)]], ignore_index=True),
            vasopressors=c_old.vasopressors,
            urine=c_old.urine,
            devices=c_old.devices,
            micro=c_old.micro,
            antimicrobials=c_old.antimicrobials,
            comorbidities=c_old.comorbidities,
            issues=c_old.issues,
        )
    else:
        merged = c_new

    write_cache(merged, cache_dir)

    # Clean up temp dir
    for p in out.glob("*.psv"):
        p.unlink()
    out.rmdir()
    tmp_dir.rmdir()

    n_stays = len(merged.cohort)
    n_pos = int(merged.cohort["has_sepsis_label"].sum())
    print(f"Batch {subset} [{start}..{end}]: downloaded={ok}/{len(batch)} | total_stays={n_stays} (sepsis_positive={n_pos})")
    return n_stays


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["training_setA", "training_setB"], required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    fetch_batch(args.subset, args.start, args.end, args.workers)
