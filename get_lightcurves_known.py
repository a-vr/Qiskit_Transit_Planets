from lightkurve import search_lightcurve
import lightkurve
import matplotlib.pyplot as plt
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import os

def get_known_curves(N):
    # Query confirmed exoplanets table
    exoplanets = NasaExoplanetArchive.query_criteria(table="pscomppars")

    # Convert to DataFrame and filter missing facilities
    df = exoplanets.to_pandas()
    df = df[df['disc_facility'].notna()]

    # Preview columns
    print(df.columns)

    # Get a list of unique host star names (Kepler, TESS, etc.)
    targets = df[['hostname', 'disc_facility']].drop_duplicates().head(N)  # Limit to N for testing
    print(targets)

    # Create output folder
    os.makedirs("lightcurves_known", exist_ok=True)

    for row in targets.itertuples(index=False):
        star = row.hostname
        facility = row.disc_facility

        # Infer mission from facility, stick to Kepler since unknowns will be from Kepler
        mission = None
        if "Kepler" in facility:
            mission = "Kepler"
        # skip other mission stars
        if not mission:
            continue

        try:
            search_result = search_lightcurve(star, mission=mission)

            # download and combine all lightcurves 
            lcfs = search_result.download_all()
            lc = lcfs.stitch().remove_nans()
            filename = f"lightcurves_known/{star.replace(' ', '_')}_{mission}_lightcurve.csv"
            lc.to_pandas().to_csv(filename, index=False)
            print(filename)

        except Exception as e:
            print(f"Failed for {star} ({mission}): {e}")

