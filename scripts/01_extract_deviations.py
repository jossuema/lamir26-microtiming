"""Extract per-position timing deviations for both corpora -> parquet.

Usage:
    python scripts/01_extract_deviations.py [--limit N] [--dataset candombe|brid|all]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings

# Targeted filters only: librosa/audioread decode chatter. A blanket ignore
# would also silence real diagnostics; per-track failures are print()ed by
# the pipeline precisely so no filter can hide them.
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)

import mirdata

from microtiming.pipeline import process_brid_mixtures, process_candombe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max tracks per dataset")
    ap.add_argument("--dataset", choices=["candombe", "brid", "all"], default="all")
    args = ap.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    if args.dataset in ("candombe", "all"):
        ds = mirdata.initialize("candombe", data_home=str(ROOT / "data/candombe"))
        df = process_candombe(ds, limit=args.limit)
        path = out_dir / "deviations_candombe.parquet"
        df.to_parquet(path, index=False)
        print(f"candombe: {len(df)} onsets from {df.track_id.nunique()} tracks -> {path}")

    if args.dataset in ("brid", "all"):
        ds = mirdata.initialize("brid", data_home=str(ROOT / "data/brid"))
        df = process_brid_mixtures(ds, limit=args.limit)
        path = out_dir / "deviations_brid.parquet"
        df.to_parquet(path, index=False)
        print(f"brid: {len(df)} onsets from {df.track_id.nunique()} tracks -> {path}")


if __name__ == "__main__":
    main()
