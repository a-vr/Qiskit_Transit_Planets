"""Unit tests for the data pipeline (preprocess + features).

All tests run on synthetic data — no internet access or Kepler downloads
are required.  Tests are designed to be fast (< 5 s each on a laptop CPU).
"""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build synthetic light curves
# ---------------------------------------------------------------------------

def _make_synthetic_lc(n: int = 500, inject_transit: bool = False):
    """Return a lightkurve LightCurve with optional injected transit dips."""
    import lightkurve as lk

    time = np.linspace(0, 30, n)
    flux = np.random.default_rng(0).normal(1.0, 0.001, n)

    if inject_transit:
        # Inject a shallow periodic dip every 5 days, depth 0.01, duration 0.1 d
        for t0 in np.arange(0, 30, 5.0):
            mask = np.abs(time - t0) < 0.05
            flux[mask] -= 0.01

    return lk.LightCurve(time=time, flux=flux)


def _synthetic_csv(tmp_path, n: int = 500, inject_transit: bool = False) -> str:
    """Save a synthetic LightCurve to a temp CSV and return its path."""
    lc = _make_synthetic_lc(n=n, inject_transit=inject_transit)
    path = str(tmp_path / "test_lc.csv")
    lc.to_pandas().to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_load_csv_returns_lightcurve(self, tmp_path):
        from src.data.preprocess import load_csv
        import lightkurve as lk

        path = _synthetic_csv(tmp_path)
        lc = load_csv(path)
        assert isinstance(lc, lk.LightCurve)

    def test_load_csv_preserves_length(self, tmp_path):
        from src.data.preprocess import load_csv

        path = _synthetic_csv(tmp_path, n=300)
        lc = load_csv(path)
        assert len(lc) == 300

    def test_load_csv_raises_on_missing_column(self, tmp_path):
        import pandas as pd
        from src.data.preprocess import load_csv

        bad_path = str(tmp_path / "bad.csv")
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_path, index=False)
        with pytest.raises(ValueError, match="flux"):
            load_csv(bad_path)

    def test_preprocess_output_shape(self, tmp_path):
        from src.data.preprocess import preprocess_csv

        path = _synthetic_csv(tmp_path, n=500)
        lc = preprocess_csv(path)
        # After remove_outliers and flatten the length may be slightly shorter
        assert len(lc) > 0

    def test_preprocess_flux_near_unity_median(self, tmp_path):
        from src.data.preprocess import preprocess_csv

        path = _synthetic_csv(tmp_path, n=500)
        lc = preprocess_csv(path)
        flux = np.asarray(lc.flux)
        # After normalization and flattening the median should be close to 1
        assert abs(np.nanmedian(flux) - 1.0) < 0.05

    def test_preprocess_csv_short_lightcurve(self, tmp_path):
        """Short light curves (< window_length) should not crash."""
        from src.data.preprocess import preprocess_csv

        path = _synthetic_csv(tmp_path, n=50)
        lc = preprocess_csv(path, window_length=401)
        assert len(lc) > 0


# ---------------------------------------------------------------------------
# feature extraction tests
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_extract_features_shape(self, tmp_path):
        from src.data.preprocess import preprocess_csv
        from src.data.features import extract_features, N_FEATURES

        path = _synthetic_csv(tmp_path, n=500)
        lc = preprocess_csv(path)
        feats = extract_features(lc)
        assert feats.shape == (N_FEATURES,)

    def test_extract_features_dtype(self, tmp_path):
        from src.data.preprocess import preprocess_csv
        from src.data.features import extract_features

        path = _synthetic_csv(tmp_path, n=500)
        lc = preprocess_csv(path)
        feats = extract_features(lc)
        assert feats.dtype == np.float64

    def test_extract_features_finite(self, tmp_path):
        from src.data.preprocess import preprocess_csv
        from src.data.features import extract_features

        path = _synthetic_csv(tmp_path, n=500)
        lc = preprocess_csv(path)
        feats = extract_features(lc)
        assert np.all(np.isfinite(feats)), f"Non-finite features: {feats}"

    def test_transit_depth_in_range(self, tmp_path):
        from src.data.preprocess import preprocess_csv
        from src.data.features import extract_features, FEATURE_NAMES

        path = _synthetic_csv(tmp_path, n=500, inject_transit=True)
        lc = preprocess_csv(path)
        feats = extract_features(lc)
        depth_idx = FEATURE_NAMES.index("transit_depth")
        assert 0.0 <= feats[depth_idx] <= 1.0, f"transit_depth out of [0,1]: {feats[depth_idx]}"

    def test_transit_duration_in_range(self, tmp_path):
        from src.data.preprocess import preprocess_csv
        from src.data.features import extract_features, FEATURE_NAMES

        path = _synthetic_csv(tmp_path, n=500, inject_transit=True)
        lc = preprocess_csv(path)
        feats = extract_features(lc)
        dur_idx = FEATURE_NAMES.index("transit_duration_frac")
        assert 0.0 <= feats[dur_idx] <= 1.0, f"transit_duration_frac out of [0,1]: {feats[dur_idx]}"


# ---------------------------------------------------------------------------
# build_dataset + stratified_split tests
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def _make_n_csvs(self, tmp_path, n: int, label_dir: str, inject: bool) -> list:
        import lightkurve as lk

        rng = np.random.default_rng(42)
        paths = []
        d = tmp_path / label_dir
        d.mkdir(exist_ok=True)
        for i in range(n):
            time = np.linspace(0, 30, 300)
            flux = rng.normal(1.0, 0.001, 300)
            if inject:
                for t0 in np.arange(0, 30, 5.0):
                    mask = np.abs(time - t0) < 0.05
                    flux[mask] -= 0.01
            p = str(d / f"star_{i}.csv")
            lk.LightCurve(time=time, flux=flux).to_pandas().to_csv(p, index=False)
            paths.append(p)
        return paths

    def test_build_dataset_shape(self, tmp_path):
        from src.data.features import build_dataset, N_FEATURES

        k_paths = self._make_n_csvs(tmp_path, 6, "known", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 6, "unknown", inject=False)
        X, y, scaler, feat_names = build_dataset(k_paths, u_paths)
        assert X.shape == (12, N_FEATURES)
        assert y.shape == (12,)

    def test_build_dataset_labels(self, tmp_path):
        from src.data.features import build_dataset

        k_paths = self._make_n_csvs(tmp_path, 4, "known2", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 4, "unknown2", inject=False)
        _, y, _, _ = build_dataset(k_paths, u_paths)
        assert set(y.tolist()) == {0, 1}
        assert int((y == 1).sum()) == 4
        assert int((y == 0).sum()) == 4

    def test_build_dataset_scaler_zero_mean(self, tmp_path):
        from src.data.features import build_dataset

        k_paths = self._make_n_csvs(tmp_path, 6, "known3", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 6, "unknown3", inject=False)
        X, _, scaler, _ = build_dataset(k_paths, u_paths)
        # StandardScaler should produce near-zero column means
        assert np.allclose(X.mean(axis=0), 0.0, atol=1e-10)

    def test_stratified_split_sizes(self, tmp_path):
        from src.data.features import build_dataset, stratified_split

        k_paths = self._make_n_csvs(tmp_path, 14, "known4", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 14, "unknown4", inject=False)
        X, y, _, _ = build_dataset(k_paths, u_paths)
        X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y)
        total = len(y_tr) + len(y_val) + len(y_te)
        assert total == len(y)

    def test_stratified_split_preserves_class_ratio(self, tmp_path):
        from src.data.features import build_dataset, stratified_split

        k_paths = self._make_n_csvs(tmp_path, 14, "known5", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 14, "unknown5", inject=False)
        X, y, _, _ = build_dataset(k_paths, u_paths)
        _, _, _, _, _, y_te = stratified_split(X, y)
        # Expect roughly 50 % positives (within 1 sample tolerance on small sets)
        ratio = y_te.mean()
        assert 0.3 <= ratio <= 0.7, f"Test set class ratio {ratio:.2f} is far from 50 %"

    def test_save_and_load_dataset(self, tmp_path):
        from src.data.features import build_dataset, save_dataset, load_dataset

        k_paths = self._make_n_csvs(tmp_path, 4, "known6", inject=True)
        u_paths = self._make_n_csvs(tmp_path, 4, "unknown6", inject=False)
        X, y, scaler, _ = build_dataset(k_paths, u_paths)

        out = str(tmp_path / "processed")
        save_dataset(X, y, scaler, out_dir=out)
        X2, y2, scaler2 = load_dataset(out_dir=out)

        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
