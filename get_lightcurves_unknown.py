import lightkurve as lk
import matplotlib.pyplot as plt
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import os
import requests
import io
import shutil
import gzip


def get_unknown_curves(N):
    # URL MAST Kepler Stellar Table
    url = "https://archive.stsci.edu/pub/kepler/catalogs/kepler_stellar_17.csv.gz"

    # Download gz
    gz_filename = "kepler_stellar_17.csv.gz"
    csv_filename = "kepler_stellar_17.csv"
    with requests.get(url, stream=True) as r:
        with open(gz_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # decompress
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(csv_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # save locally (will overwrite previous)
    kepler_df = pd.read_csv(csv_filename)
    print(f"Loaded {len(kepler_df)} rows.")
    print(kepler_df.head())

    # Load confirmed planets
    confirmed = NasaExoplanetArchive.query_criteria(table="pscomppars").to_pandas()
    known_hosts = set(confirmed['hostname'])

    # Load a list of KICs from MAST Kepler Stellar Table
    kic_list = pd.read_csv("kepler_stellar_17.csv") 
    kic_list = kic_list[~kic_list['kepid'].isin(known_hosts)]

    # Create output folder
    os.makedirs("lightcurves_unknown", exist_ok=True)

    # Get light curves for N random non-planet stars
    sample = kic_list.sample(N, random_state=42)
    for kic in sample['kepid']:
        star = f'KIC {kic}'
        try: 
            search_result = lk.search_lightcurve(star, mission='Kepler')
            lcfs = search_result.download_all()
            lc = lcfs.stitch().remove_nans()
            filename = f"lightcurves_unknown/{star.replace(' ', '_')}_Kepler_lightcurve.csv"
            lc.to_pandas().to_csv(filename, index=False)
            print(filename)
        except Exception as e:
            print(f"Failed for {star} (Kepler): {e}")




