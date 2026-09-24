"""Preprocess raw Kepler light curves: load, clean, detrend, and normalize.

The core preprocessing steps are:
    1. Remove NaN cadences and sigma-clip outliers.
    2. Normalize flux to unit median (``normalize()``).
    3. Detrend stellar variability with a Savitzky-Golay filter (``flatten()``).

The output is a ``lightkurve.LightCurve`` ready for feature extraction.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: str):
    """Load a lightkurve-exported CSV back into a LightCurve object.

    lightkurve exports columns such as ``time``, ``flux``, and optionally
    ``flux_err``.  The column names are matched case-insensitively.

    Args:
        path: Path to a CSV previously saved by ``LightCurve.to_pandas().to_csv()``.

    Returns:
        A ``lightkurve.LightCurve`` object.

    Raises:
        ImportError: If lightkurve is not installed.
        ValueError: If the CSV is missing required columns.
        FileNotFoundError: If *path* does not exist.
    """
    import lightkurve as lk  # deferred so tests can mock

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    time_key = next((cols_lower[k] for k in cols_lower if "time" in k), None)
    flux_key = next((cols_lower[k] for k in cols_lower if k == "flux"), None)

    if time_key is None or flux_key is None:
        raise ValueError(
            f"{path}: expected columns 'time' and 'flux', found {list(df.columns)}"
        )

    err_key = next((cols_lower[k] for k in cols_lower if "flux_err" in k), None)
    flux_err = df[err_key].values if err_key else None

    return lk.LightCurve(
        time=df[time_key].values,
        flux=df[flux_key].values,
        flux_err=flux_err,
    )


def preprocess(lc, window_length: int = 401, sigma_outlier: float = 5.0):
    """Normalize and detrend a light curve.

    Steps:
        1. Remove NaN cadences.
        2. Remove outliers beyond *sigma_outlier* standard deviations.
        3. Normalize so the median flux equals 1.
        4. Flatten using a Savitzky-Golay filter (removes stellar variability
           on timescales longer than *window_length* cadences).

    Args:
        lc: A ``lightkurve.LightCurve`` object.
        window_length: Window length for the Savitzky-Golay filter (must be odd).
            At Kepler's 30-min long cadence, 401 cadences ≈ 8.4 days, which
            suppresses most stellar rotation and instrumental trends while
            preserving transit-depth signals that last hours.
        sigma_outlier: Rejection threshold for outlier removal.

    Returns:
        Cleaned, normalized, detrended ``lightkurve.LightCurve``.
    """
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=sigma_outlier)
    lc = lc.normalize()
    # Ensure window_length is odd and no larger than the light curve
    n = len(lc)
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < 3:
        return lc  # too short to flatten; return as-is
    lc = lc.flatten(window_length=wl, polyorder=2)
    return lc


def preprocess_csv(path: str, window_length: int = 401, sigma_outlier: float = 5.0):
    """Load a CSV and return a preprocessed LightCurve.

    Convenience wrapper around :func:`load_csv` + :func:`preprocess`.

    Args:
        path: Path to a lightkurve-exported CSV.
        window_length: Passed to :func:`preprocess`.
        sigma_outlier: Passed to :func:`preprocess`.

    Returns:
        Preprocessed ``lightkurve.LightCurve``.
    """
    lc = load_csv(path)
    return preprocess(lc, window_length=window_length, sigma_outlier=sigma_outlier)
