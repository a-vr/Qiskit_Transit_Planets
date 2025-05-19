# This file checks targets without known transitting planets
# Can be used to find new exoplanets

import argparse
from get_lightcurves_unknown import get_unknown_curves
from normalize_curves import normalize_lightcurves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download known Kepler exoplanet lightcurves.")
    parser.add_argument("--n", type=int, default=10, help="Number of known stars to download")
    args = parser.parse_args()

    get_unknown_curves(N=args.n)
    normalize_lightcurves("lightcurves_unknown")

