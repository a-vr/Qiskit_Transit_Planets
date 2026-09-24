"""Smoke tests for QKEModel and VQCModel.

All tests use small synthetic datasets (10 training + 4 test samples, 8
features) and minimal iteration counts to keep runtime under ~60 s per test
on a laptop CPU.  No real Kepler data is downloaded.

Qiskit packages are imported lazily via ``pytest.importorskip`` so the file
can be collected without them installed.
"""

import os
import tempfile

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
qml = pytest.importorskip(
    "qiskit_machine_learning", reason="qiskit-machine-learning not installed"
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_dataset():
    """10 training + 4 test samples, balanced binary labels, 8 features."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((10, 8))
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X_test = rng.standard_normal((4, 8))
    y_test = np.array([0, 0, 1, 1])
    return X_train, y_train, X_test, y_test


# ── QKEModel tests ─────────────────────────────────────────────────────────────

class TestQKEModel:
    def test_fit_returns_self(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = QKEModel(C=1.0)
        result = model.fit(X_tr, y_tr)
        assert result is model

    def test_train_time_recorded(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        assert model.train_time_s is not None
        assert model.train_time_s > 0

    def test_predict_shape(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_only_valid_labels(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds.tolist()).issubset({0, 1})

    def test_score_in_unit_interval(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, y_te = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        assert 0.0 <= acc <= 1.0

    def test_decision_score_shape(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        scores = model.decision_score(X_te)
        assert scores.shape == (len(X_te),)

    def test_decision_score_dtype(self, tiny_dataset):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)
        scores = model.decision_score(X_te)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_predict_before_fit_raises(self):
        from src.quantum.model import QKEModel

        with pytest.raises(RuntimeError, match="not fitted"):
            QKEModel().predict(np.zeros((2, 8)))

    def test_save_and_load(self, tiny_dataset, tmp_path):
        from src.quantum.model import QKEModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = QKEModel().fit(X_tr, y_tr)

        save_path = str(tmp_path / "qke.joblib")
        model.save(save_path)
        assert os.path.exists(save_path)

        loaded = QKEModel.load(save_path)
        preds_orig = model.predict(X_te)
        preds_load = loaded.predict(X_te)
        np.testing.assert_array_equal(preds_orig, preds_load)


# ── VQCModel tests ─────────────────────────────────────────────────────────────

class TestVQCModel:
    """Tests use max_iter=50 — scipy COBYLA enforces maxfun >= num_vars+2 = 34,
    so values below 34 are silently raised; 50 keeps runtime short while
    staying above the floor and making upper-bound assertions valid."""

    _MAX_ITER = 50

    def test_fit_returns_self(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert isinstance(model, VQCModel)

    def test_loss_history_populated(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert len(model.loss_history) > 0
        # COBYLA may run more than max_iter evals when it corrects a too-small
        # budget; allow up to 2× to remain a meaningful upper bound.
        assert len(model.loss_history) <= self._MAX_ITER * 2

    def test_loss_history_all_finite(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert all(np.isfinite(v) for v in model.loss_history)

    def test_loss_history_positive(self, tiny_dataset):
        """Cross-entropy loss must be non-negative."""
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert all(v >= 0 for v in model.loss_history)

    def test_train_time_recorded(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert model.train_time_s is not None and model.train_time_s > 0

    def test_predict_shape(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_predict_only_valid_labels(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds.tolist()).issubset({0, 1})

    def test_score_in_unit_interval(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, X_te, y_te = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        assert 0.0 <= acc <= 1.0

    def test_decision_score_shape(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        scores = model.decision_score(X_te)
        assert scores.shape == (len(X_te),)

    def test_weights_shape(self, tiny_dataset):
        from src.quantum.model import VQCModel
        from src.quantum.feature_map import N_QUBITS, RA_REPS

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        assert model.weights.shape == (N_QUBITS * (RA_REPS + 1),)

    def test_predict_before_fit_raises(self):
        from src.quantum.model import VQCModel

        with pytest.raises(RuntimeError, match="not fitted"):
            VQCModel().predict(np.zeros((2, 8)))

    def test_convergence_summary_keys(self, tiny_dataset):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        summary = model.convergence_summary()
        assert {"n_evals", "initial_loss", "final_loss", "loss_reduction", "converged"}.issubset(
            set(summary.keys())
        )

    def test_save_creates_files(self, tiny_dataset, tmp_path):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        base = str(tmp_path / "vqc")
        model.save(base)
        assert os.path.exists(base + ".weights.npy")
        assert os.path.exists(base + ".meta.json")

    def test_load_restores_weights_and_history(self, tiny_dataset, tmp_path):
        from src.quantum.model import VQCModel

        X_tr, y_tr, _, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        base = str(tmp_path / "vqc2")
        model.save(base)

        loaded = VQCModel.load(base)
        np.testing.assert_array_almost_equal(loaded.initial_point, model.weights)
        assert loaded.loss_history == model.loss_history
        assert loaded.max_iter == model.max_iter

    def test_warm_start_from_loaded_weights(self, tiny_dataset, tmp_path):
        """A model loaded from disk warm-starts correctly and can predict."""
        from src.quantum.model import VQCModel

        X_tr, y_tr, X_te, _ = tiny_dataset
        model = VQCModel(max_iter=self._MAX_ITER).fit(X_tr, y_tr)
        base = str(tmp_path / "vqc3")
        model.save(base)

        warm = VQCModel.load(base)
        warm.fit(X_tr, y_tr)  # warm-start with max_iter iterations from saved point
        preds = warm.predict(X_te)
        assert preds.shape == (len(X_te),)
