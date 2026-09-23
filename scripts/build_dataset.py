"""Build the full feature dataset from Kepler light curves.

Usage
-----
    python scripts/build_dataset.py --known 80 --unknown 80

Downloads light curves (skips files already cached), preprocesses them,
extracts 8 features per star, and saves the dataset to ``data/processed/``.

Run this script once before training.  The download step requires internet
access to MAST (via lightkurve) and the NASA Exoplanet Archive.
"""

import argparse
import logging
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download import download_known_stars, download_unknown_stars
from src.data.features import build_dataset, save_dataset, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Download and featurize Kepler light curves.")
    parser.add_argument("--known", type=int, default=80,
                        help="Number of confirmed planet-host stars (default: 80)")
    parser.add_argument("--unknown", type=int, default=80,
                        help="Number of non-host stars (default: 80)")
    parser.add_argument("--known-dir", default="data/raw/known",
                        help="Directory for known-star CSV files")
    parser.add_argument("--unknown-dir", default="data/raw/unknown",
                        help="Directory for unknown-star CSV files")
    parser.add_argument("--out-dir", default="data/processed",
                        help="Output directory for features.npy / labels.npy")
    args = parser.parse_args()

    print(f"Downloading {args.known} known-planet stars…")
    known_paths = download_known_stars(args.known, output_dir=args.known_dir)

    print(f"Downloading {args.unknown} non-planet stars…")
    unknown_paths = download_unknown_stars(args.unknown, output_dir=args.unknown_dir)

    print(f"Extracting features from {len(known_paths)} known + {len(unknown_paths)} unknown stars…")
    X, y, scaler, _ = build_dataset(known_paths, unknown_paths)

    pos = int(y.sum())
    neg = int((y == 0).sum())
    print(f"Dataset: {len(y)} stars  |  {pos} planet hosts  |  {neg} non-hosts")
    print(f"Features: {FEATURE_NAMES}")
    print(f"Class ratio (positive / total): {pos / len(y):.2%}")

    save_dataset(X, y, scaler, out_dir=args.out_dir)
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
