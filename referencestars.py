# This file works like main except for targets with known transitting planets
# Can be used to verify identification algorithm

import argparse
from get_lightcurves_known import get_known_curves
from normalize_curves import normalize_lightcurves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download known Kepler exoplanet lightcurves.")
    parser.add_argument("--n", type=int, default=50, help="Number of known stars to download")
    args = parser.parse_args()

    get_known_curves(N=args.n)
    normalize_lightcurves("lightcurves_known")