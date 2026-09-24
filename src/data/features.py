"""Extract an 8-dimensional feature vector from each Kepler light curve.

Feature design rationale
------------------------
A quantum circuit with 8 qubits can encode at most 8 independent input
features when using angle/amplitude encoding (ZZFeatureMap).  We therefore
extract exactly 8 physically meaningful features rather than applying blind
PCA to a 500-point raw flux vector.

The 8 features are derived from a Box Least-Squares (BLS) periodogram plus
basic flux statistics:

    1. bls_peak_power     – Normalized peak BLS power (strength of best
                            periodic signal).  High values indicate a
                            coherent, repeating flux dip consistent with a
                            transit.
    2. transit_depth      – Fractional flux drop at the best-fit period (0–1).
                            Deep dips are more likely transits.
    3. transit_duration   – Duration of the transit as a fraction of the
                            best-fit period (0–1).  Real planetary transits
                            are typically 1–10 % of the orbital period.
    4. log_period         – log₁₀ of the best-fit BLS period in days.
                            Kepler planets range from ~0.5 to 500 days;
                            log-scaling compresses this range.
    5. transit_snr        – Transit SNR = depth / out-of-transit scatter.
                            Separates deep transits in noisy stars from
                            shallow dips in quiet stars.
    6. rms_scatter        – RMS of the normalized flux over the full light
                            curve.  Proxy for photometric noise level and
                            stellar activity.
    7. flux_skewness      – Third standardized moment of the flux distribution.
                            Transits pull flux below the median, producing
                            negative skewness in transit hosts.
    8. flux_kurtosis      – Excess kurtosis of the flux distribution.  Transit
                            events create sharp downward spikes, increasing
                            kurtosis relative to a pure Gaussian baseline.

All features are scaled to zero mean / unit variance via a fitted
``StandardScaler`` before being passed to the quantum circuit.
"""

import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bls_peak_power",
    "transit_depth",
    "transit_duration_frac",
    "log_period_days",
    "transit_snr",
    "rms_scatter",
    "flux_skewness",
    "flux_kurtosis",
]
N_FEATURES = 8  # number of qubits == number of features

# BLS period search range (days)
_BLS_PERIOD_MIN = 0.5
_BLS_PERIOD_MAX = 30.0
_BLS_PERIOD_STEPS = 500


def _standardized_moment(x: np.ndarray, order: int) -> float:
    """Return the *order*-th standardized central moment of *x*."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0.0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** order))


def extract_features(lc) -> np.ndarray:
    """Extract the 8-element feature vector from a preprocessed light curve.

    Args:
        lc: A preprocessed ``lightkurve.LightCurve`` (output of
            :func:`src.data.preprocess.preprocess`).

    Returns:
        ``np.ndarray`` of shape ``(8,)`` with dtype ``float64``.
        On BLS failure, BLS-derived features default to 0.
    """
    flux = np.asarray(lc.flux, dtype=float)

    # -----------------------------------------------------------------------
    # BLS periodogram
    # -----------------------------------------------------------------------
    bls_power = 0.0
    transit_depth = 0.0
    transit_duration_frac = 0.0
    log_period = 0.0
    transit_snr = 0.0

    try:
        period_grid = np.linspace(_BLS_PERIOD_MIN, _BLS_PERIOD_MAX, _BLS_PERIOD_STEPS)
        pg = lc.to_periodogram(method="bls", period=period_grid)

        # Peak power (normalized by mean to make it scale-invariant)
        powers = np.asarray(pg.power)
        max_power = float(np.nanmax(powers))
        mean_power = float(np.nanmean(powers))
        bls_power = (max_power / mean_power) if mean_power > 0 else 0.0

        # Best-period parameters
        best_period = float(pg.period_at_max_power.to("d").value)
        depth_qty = pg.depth_at_max_power
        duration_qty = pg.duration_at_max_power

        transit_depth = float(depth_qty) if depth_qty is not None else 0.0
        transit_depth = max(0.0, min(transit_depth, 1.0))  # clamp to [0, 1]

        duration_days = float(duration_qty.to("d").value) if duration_qty is not None else 0.0
        transit_duration_frac = (
            duration_days / best_period if best_period > 0 else 0.0
        )
        transit_duration_frac = max(0.0, min(transit_duration_frac, 1.0))

        log_period = np.log10(best_period) if best_period > 0 else 0.0

        # Out-of-transit scatter for SNR
        folded = lc.fold(period=best_period)
        f_folded = np.asarray(folded.flux, dtype=float)
        depth_thresh = 1.0 - transit_depth / 2.0
        oot_mask = f_folded > depth_thresh
        oot_scatter = float(np.nanstd(f_folded[oot_mask])) if oot_mask.sum() > 0 else float(np.nanstd(f_folded))
        transit_snr = (transit_depth / oot_scatter) if oot_scatter > 0 else 0.0

    except Exception as exc:
        logger.debug("BLS failed: %s", exc)

    # -----------------------------------------------------------------------
    # Flux statistics (always available)
    # -----------------------------------------------------------------------
    rms_scatter = float(np.nanstd(flux))
    flux_skewness = _standardized_moment(flux, 3)
    flux_kurtosis = _standardized_moment(flux, 4) - 3.0  # excess kurtosis

    return np.array(
        [
            bls_power,
            transit_depth,
            transit_duration_frac,
            log_period,
            transit_snr,
            rms_scatter,
            flux_skewness,
            flux_kurtosis,
        ],
        dtype=np.float64,
    )


def build_dataset(
    known_paths: list,
    unknown_paths: list,
    scaler: StandardScaler = None,
) -> tuple:
    """Build (X, y) arrays from lists of CSV file paths.

    Iterates over known-planet (label=1) and unknown-planet (label=0) CSV
    files, preprocessing and extracting features from each.  Files that fail
    to process are skipped with a warning.

    Class imbalance note: The dataset is typically imbalanced (far fewer
    confirmed hosts than non-hosts).  Pass ``class_weight='balanced'`` to
    downstream classifiers rather than upsampling here, so the test set
    remains an unbiased reflection of the natural class ratio.

    Args:
        known_paths: CSV paths for confirmed Kepler planet hosts (label = 1).
        unknown_paths: CSV paths for non-host Kepler stars (label = 0).
        scaler: Pre-fitted ``StandardScaler``.  If ``None``, a new scaler is
            fitted on the combined dataset and returned.

    Returns:
        Tuple of ``(X, y, scaler, feature_names)`` where:
            - X: ``np.ndarray`` of shape ``(N, 8)``.
            - y: ``np.ndarray`` of shape ``(N,)`` with 0/1 labels.
            - scaler: Fitted ``StandardScaler`` (reuse when transforming
              new data at inference time).
            - feature_names: ``list[str]`` matching columns of X.
    """
    from src.data.preprocess import preprocess_csv  # avoid circular at module level

    rows, labels = [], []

    for path in known_paths:
        try:
            lc = preprocess_csv(path)
            rows.append(extract_features(lc))
            labels.append(1)
        except Exception as exc:
            logger.warning("Skipping known star %s: %s", path, exc)

    for path in unknown_paths:
        try:
            lc = preprocess_csv(path)
            rows.append(extract_features(lc))
            labels.append(0)
        except Exception as exc:
            logger.warning("Skipping unknown star %s: %s", path, exc)

    if not rows:
        raise ValueError("No light curves were successfully processed.")

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=int)

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, scaler, FEATURE_NAMES


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Stratified train / val / test split preserving class ratios.

    Args:
        X: Feature matrix of shape ``(N, 8)``.
        y: Labels of shape ``(N,)``.
        val_frac: Fraction of total data reserved for validation.
        test_frac: Fraction of total data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``
    """
    hold_out = val_frac + test_frac
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=hold_out, stratify=y, random_state=random_state
    )
    val_relative = val_frac / hold_out
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_relative),
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_dataset(X: np.ndarray, y: np.ndarray, scaler: StandardScaler,
                 out_dir: str = "data/processed") -> None:
    """Persist feature matrix, labels, and scaler to disk.

    Args:
        X: Feature matrix.
        y: Label vector.
        scaler: Fitted StandardScaler.
        out_dir: Directory to save into.
    """
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "features.npy"), X)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    logger.info("Dataset saved to %s (N=%d, d=%d)", out_dir, len(y), X.shape[1])


def load_dataset(out_dir: str = "data/processed") -> tuple:
    """Load a previously saved feature matrix, labels, and scaler.

    Args:
        out_dir: Directory containing ``features.npy``, ``labels.npy``,
            and ``scaler.joblib``.

    Returns:
        ``(X, y, scaler)``
    """
    import joblib
    X = np.load(os.path.join(out_dir, "features.npy"))
    y = np.load(os.path.join(out_dir, "labels.npy"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    return X, y, scaler
