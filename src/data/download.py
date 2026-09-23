"""Download Kepler light curves from MAST via lightkurve.

Provides two functions:
    download_known_stars  -- confirmed Kepler planet hosts (label = 1)
    download_unknown_stars -- Kepler stars with no confirmed planets (label = 0)
"""

import gzip
import logging
import os
import shutil

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

logger = logging.getLogger(__name__)

_KEPLER_STELLAR_URL = (
    "https://archive.stsci.edu/pub/kepler/catalogs/kepler_stellar_17.csv.gz"
)
_CATALOG_CACHE = os.path.join("data", "kepler_stellar_17.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _confirmed_kepler_kepids() -> set:
    """Return the set of integer KIC IDs for confirmed Kepler planet hosts."""
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="hostname,kepid,disc_facility",
    ).to_pandas()
    kepler_rows = table[
        table["disc_facility"].fillna("").str.contains("Kepler", case=False)
        & table["kepid"].notna()
    ]
    return set(kepler_rows["kepid"].astype(int).tolist())


def _kepler_stellar_catalog(cache_path: str = _CATALOG_CACHE) -> pd.DataFrame:
    """Download and cache the MAST Kepler Stellar Catalog 17.

    Args:
        cache_path: Local path to cache the decompressed CSV.

    Returns:
        DataFrame with one row per Kepler target; guaranteed to have a
        ``kepid`` column containing integer KIC IDs.
    """
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if not os.path.exists(cache_path):
        logger.info("Downloading Kepler Stellar Catalog 17 (~60 MB)…")
        gz_path = cache_path + ".gz"
        with requests.get(_KEPLER_STELLAR_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as fout:
                shutil.copyfileobj(r.raw, fout)
        with gzip.open(gz_path, "rb") as f_in, open(cache_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        logger.info("Catalog saved to %s", cache_path)

    df = pd.read_csv(cache_path, comment="#")
    # Normalize column name: the catalog uses 'kepid' but may have leading spaces
    df.columns = [c.strip() for c in df.columns]
    if "kepid" not in df.columns:
        kepid_col = next((c for c in df.columns if "kepid" in c.lower()), None)
        if kepid_col is None:
            raise ValueError(
                f"Cannot locate kepid column in catalog. Columns: {list(df.columns)}"
            )
        df = df.rename(columns={kepid_col: "kepid"})
    df["kepid"] = df["kepid"].astype(int)
    return df


def _save_lightcurve(star_id: str, out_path: str) -> bool:
    """Download, stitch, and save all Kepler long-cadence quarters for *star_id*.

    Args:
        star_id: Lightkurve target string, e.g. ``"KIC 757450"``.
        out_path: Destination CSV path.

    Returns:
        True on success, False if the star has no data or download fails.
    """
    try:
        result = lk.search_lightcurve(star_id, mission="Kepler", exptime="long")
        if len(result) == 0:
            logger.debug("No Kepler data found for %s", star_id)
            return False
        collection = result.download_all()
        if collection is None or len(collection) == 0:
            return False
        lc = collection.stitch().remove_nans()
        lc.to_pandas().to_csv(out_path, index=False)
        return True
    except Exception as exc:
        logger.warning("Failed to download %s: %s", star_id, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_known_stars(n: int, output_dir: str = "data/raw/known") -> list:
    """Download light curves for *n* confirmed Kepler planet host stars.

    Stars are drawn from the NASA Exoplanet Archive (pscomppars table),
    filtered to Kepler discoveries with a valid KIC ID, then deduplicated
    by host name.

    Args:
        n: Maximum number of stars to download.
        output_dir: Directory where per-star CSV files are saved.

    Returns:
        List of file paths for successfully downloaded stars.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Querying NASA Exoplanet Archive for confirmed Kepler hosts…")
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="hostname,kepid,disc_facility",
    ).to_pandas()

    kepler = table[
        table["disc_facility"].fillna("").str.contains("Kepler", case=False)
        & table["kepid"].notna()
    ].drop_duplicates(subset="hostname").head(n)

    paths = []
    for _, row in kepler.iterrows():
        kepid = int(row["kepid"])
        star_id = f"KIC {kepid}"
        out_path = os.path.join(output_dir, f"KIC_{kepid}.csv")
        if os.path.exists(out_path):
            logger.debug("Already downloaded %s", star_id)
            paths.append(out_path)
            continue
        if _save_lightcurve(star_id, out_path):
            logger.info("Saved %s → %s", star_id, out_path)
            paths.append(out_path)

    logger.info("Downloaded %d / %d known-planet stars", len(paths), len(kepler))
    return paths


def download_unknown_stars(
    n: int,
    output_dir: str = "data/raw/unknown",
    catalog_cache: str = _CATALOG_CACHE,
    random_state: int = 42,
) -> list:
    """Download light curves for *n* Kepler stars without confirmed transiting planets.

    The Kepler Stellar Catalog 17 is used as the pool; confirmed planet hosts
    (matched by integer KIC ID from the NASA Exoplanet Archive) are removed
    before sampling.

    Args:
        n: Maximum number of stars to download.
        output_dir: Directory where per-star CSV files are saved.
        catalog_cache: Path to cache the Kepler stellar catalog CSV locally.
        random_state: Random seed for reproducible sampling.

    Returns:
        List of file paths for successfully downloaded stars.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Fetching confirmed Kepler KIC IDs to exclude…")
    known_kepids = _confirmed_kepler_kepids()
    logger.info("Excluding %d confirmed-host KIC IDs", len(known_kepids))

    catalog = _kepler_stellar_catalog(catalog_cache)
    unknown = catalog[~catalog["kepid"].isin(known_kepids)]

    # Oversample by 3× to tolerate download failures
    candidate_n = min(n * 3, len(unknown))
    sample = unknown.sample(candidate_n, random_state=random_state)

    paths = []
    for kepid in sample["kepid"]:
        if len(paths) >= n:
            break
        star_id = f"KIC {kepid}"
        out_path = os.path.join(output_dir, f"KIC_{kepid}.csv")
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        if _save_lightcurve(star_id, out_path):
            logger.info("Saved %s → %s", star_id, out_path)
            paths.append(out_path)

    logger.info("Downloaded %d / %d unknown-planet stars", len(paths), n)
    return paths
