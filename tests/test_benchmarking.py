"""Tests for classical baselines and benchmarking utilities.

Classical model tests do not require Qiskit and run without any special
marks.  Benchmarking tests only need sklearn + pandas + scipy + matplotlib,
all of which are in requirements.txt.
"""

import os
import tempfile

import numpy as np
import pytest


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_dataset():
    """20 training + 6 test samples, balanced, 8 features."""
    rng = np.random.default_rng(7)
    X_tr = rng.standard_normal((20, 8))
    y_tr = np.array([0] * 10 + [1] * 10)
    X_te = rng.standard_normal((6, 8))
    y_te = np.array([0, 0, 0, 1, 1, 1])
    return X_tr, y_tr, X_te, y_te


# ── SVMBaseline ────────────────────────────────────────────────────────────────

class TestSVMBaseline:
    def test_fit_returns_self(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, _, _ = tiny_dataset
        m = SVMBaseline()
        assert m.fit(X_tr, y_tr) is m

    def test_predict_shape(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        preds = SVMBaseline().fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_valid_labels(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        preds = SVMBaseline().fit(X_tr, y_tr).predict(X_te)
        assert set(preds.tolist()).issubset({0, 1})

    def test_decision_score_shape(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        scores = SVMBaseline().fit(X_tr, y_tr).decision_score(X_te)
        assert scores.shape == (len(X_te),)

    def test_score_in_unit_interval(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, y_te = tiny_dataset
        acc = SVMBaseline().fit(X_tr, y_tr).score(X_te, y_te)
        assert 0.0 <= acc <= 1.0

    def test_train_time_recorded(self, tiny_dataset):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, _, _ = tiny_dataset
        m = SVMBaseline().fit(X_tr, y_tr)
        assert m.train_time_s is not None and m.train_time_s >= 0

    def test_predict_before_fit_raises(self):
        from src.classical.baseline import SVMBaseline
        with pytest.raises(RuntimeError, match="not fitted"):
            SVMBaseline().predict(np.zeros((2, 8)))

    def test_save_and_load_roundtrip(self, tiny_dataset, tmp_path):
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        m = SVMBaseline().fit(X_tr, y_tr)
        path = str(tmp_path / "svm.joblib")
        m.save(path)
        loaded = SVMBaseline.load(path)
        np.testing.assert_array_equal(m.predict(X_te), loaded.predict(X_te))


# ── LogisticBaseline ───────────────────────────────────────────────────────────

class TestLogisticBaseline:
    def test_fit_returns_self(self, tiny_dataset):
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, _, _ = tiny_dataset
        assert LogisticBaseline().fit(X_tr, y_tr) is LogisticBaseline().fit(X_tr, y_tr).__class__()  # noqa — just check it runs
        m = LogisticBaseline()
        assert m.fit(X_tr, y_tr) is m

    def test_predict_shape(self, tiny_dataset):
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        preds = LogisticBaseline().fit(X_tr, y_tr).predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_decision_score_in_unit_interval(self, tiny_dataset):
        """predict_proba output must lie in [0, 1]."""
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        scores = LogisticBaseline().fit(X_tr, y_tr).decision_score(X_te)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_score_in_unit_interval(self, tiny_dataset):
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, X_te, y_te = tiny_dataset
        acc = LogisticBaseline().fit(X_tr, y_tr).score(X_te, y_te)
        assert 0.0 <= acc <= 1.0

    def test_train_time_recorded(self, tiny_dataset):
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, _, _ = tiny_dataset
        m = LogisticBaseline().fit(X_tr, y_tr)
        assert m.train_time_s is not None and m.train_time_s >= 0

    def test_predict_before_fit_raises(self):
        from src.classical.baseline import LogisticBaseline
        with pytest.raises(RuntimeError, match="not fitted"):
            LogisticBaseline().predict(np.zeros((2, 8)))

    def test_save_and_load_roundtrip(self, tiny_dataset, tmp_path):
        from src.classical.baseline import LogisticBaseline
        X_tr, y_tr, X_te, _ = tiny_dataset
        m = LogisticBaseline().fit(X_tr, y_tr)
        path = str(tmp_path / "lr.joblib")
        m.save(path)
        loaded = LogisticBaseline.load(path)
        np.testing.assert_array_equal(m.predict(X_te), loaded.predict(X_te))


# ── compare.evaluate_model ─────────────────────────────────────────────────────

class TestEvaluateModel:
    def _fitted_svm(self, X_tr, y_tr):
        from src.classical.baseline import SVMBaseline
        return SVMBaseline().fit(X_tr, y_tr)

    def test_returns_expected_keys(self, tiny_dataset):
        from src.benchmarking.compare import evaluate_model
        X_tr, y_tr, X_te, y_te = tiny_dataset
        m = self._fitted_svm(X_tr, y_tr)
        result = evaluate_model(m, X_te, y_te, "SVM")
        required = {"model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"}
        assert required.issubset(set(result.keys()))

    def test_accuracy_in_unit_interval(self, tiny_dataset):
        from src.benchmarking.compare import evaluate_model
        X_tr, y_tr, X_te, y_te = tiny_dataset
        m = self._fitted_svm(X_tr, y_tr)
        r = evaluate_model(m, X_te, y_te, "SVM")
        assert 0.0 <= r["accuracy"] <= 1.0

    def test_roc_auc_in_unit_interval(self, tiny_dataset):
        from src.benchmarking.compare import evaluate_model
        X_tr, y_tr, X_te, y_te = tiny_dataset
        m = self._fitted_svm(X_tr, y_tr)
        r = evaluate_model(m, X_te, y_te, "SVM")
        assert r["roc_auc"] is None or 0.0 <= r["roc_auc"] <= 1.0

    def test_train_time_forwarded(self, tiny_dataset):
        from src.benchmarking.compare import evaluate_model
        X_tr, y_tr, X_te, y_te = tiny_dataset
        m = self._fitted_svm(X_tr, y_tr)
        r = evaluate_model(m, X_te, y_te, "SVM")
        assert r["train_time_s"] == m.train_time_s


class TestCompareAll:
    def test_dataframe_rows_match_models(self, tiny_dataset):
        import pandas as pd
        from src.benchmarking.compare import compare_all
        from src.classical.baseline import SVMBaseline, LogisticBaseline

        X_tr, y_tr, X_te, y_te = tiny_dataset
        models = {
            "SVM": SVMBaseline().fit(X_tr, y_tr),
            "LR":  LogisticBaseline().fit(X_tr, y_tr),
        }
        df = compare_all(models, X_te, y_te)
        assert isinstance(df, pd.DataFrame)
        assert set(df.index.tolist()) == {"SVM", "LR"}

    def test_dataframe_has_metric_columns(self, tiny_dataset):
        from src.benchmarking.compare import compare_all
        from src.classical.baseline import SVMBaseline

        X_tr, y_tr, X_te, y_te = tiny_dataset
        df = compare_all({"SVM": SVMBaseline().fit(X_tr, y_tr)}, X_te, y_te)
        for col in ("accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"):
            assert col in df.columns


# ── Plotting smoke tests ───────────────────────────────────────────────────────

class TestPlots:
    """Test that plot functions produce a file without crashing.
    Only checks file creation, not visual correctness."""

    def _models(self, X_tr, y_tr):
        from src.classical.baseline import SVMBaseline, LogisticBaseline
        return {
            "SVM (RBF)": SVMBaseline().fit(X_tr, y_tr),
            "Logistic Reg.": LogisticBaseline().fit(X_tr, y_tr),
        }

    def test_plot_roc_curves_saves_file(self, tiny_dataset, tmp_path):
        from src.benchmarking.compare import plot_roc_curves
        X_tr, y_tr, X_te, y_te = tiny_dataset
        out = str(tmp_path / "roc.png")
        plot_roc_curves(self._models(X_tr, y_tr), X_te, y_te, save_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_plot_loss_curve_saves_file(self, tmp_path):
        from src.benchmarking.compare import plot_loss_curve

        class _FakeVQC:
            loss_history = [0.8, 0.75, 0.7, 0.65, 0.62, 0.60, 0.59, 0.58]
            train_time_s = 1.0

        out = str(tmp_path / "loss.png")
        plot_loss_curve(_FakeVQC(), save_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_plot_feature_importance_saves_file(self, tiny_dataset, tmp_path):
        from src.benchmarking.compare import plot_feature_importance
        from src.data.features import FEATURE_NAMES
        X_tr, y_tr, _, _ = tiny_dataset
        out = str(tmp_path / "feat.png")
        plot_feature_importance(X_tr, y_tr, FEATURE_NAMES, save_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0

    def test_save_table_csv(self, tiny_dataset, tmp_path):
        from src.benchmarking.compare import compare_all, save_table_csv
        from src.classical.baseline import SVMBaseline
        X_tr, y_tr, X_te, y_te = tiny_dataset
        df = compare_all({"SVM": SVMBaseline().fit(X_tr, y_tr)}, X_te, y_te)
        out = str(tmp_path / "table.csv")
        save_table_csv(df, out)
        assert os.path.exists(out)
        import pandas as pd
        loaded = pd.read_csv(out, index_col=0)
        assert "accuracy" in loaded.columns
