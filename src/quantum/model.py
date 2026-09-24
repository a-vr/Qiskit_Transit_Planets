"""Training, inference, and persistence for the transit classifiers.

Two quantum models are provided:

QKEModel
    Wraps QSVC (Quantum Support Vector Classifier) with a
    FidelityQuantumKernel built from a ZZFeatureMap.  No gradient
    computation is needed; the kernel matrix is computed once in O(N²)
    circuit evaluations and the SVM dual is solved classically.

VQCModel
    Wraps VQC (Variational Quantum Classifier) with ZZFeatureMap +
    RealAmplitudes ansatz and COBYLA gradient-free optimizer.  Records the
    cross-entropy loss at every optimizer callback for convergence analysis.

Both models expose a unified interface:
    fit(X, y) → self
    predict(X) → np.ndarray of integer labels
    decision_score(X) → np.ndarray of real-valued scores for ROC-AUC
    score(X, y) → float accuracy
    save(path) / load(path)
"""

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


# ── QKE ───────────────────────────────────────────────────────────────────────

class QKEModel:
    """QSVC with the project's FidelityQuantumKernel (ZZFeatureMap, reps=2).

    Training complexity: O(N²) kernel evaluations + classical SVM solve.
    For N=60 training samples and 8 qubits, expect ~30–90 s on a laptop CPU.

    Attributes:
        C: SVM regularization parameter.
        class_weight: Passed to QSVC; ``'balanced'`` adjusts for class
            imbalance by weighting minority-class errors more heavily.
        train_time_s: Wall-clock training time in seconds (set after fit).
    """

    def __init__(self, C: float = 1.0, class_weight: str = "balanced"):
        self.C = C
        self.class_weight = class_weight
        self._qsvc = None
        self.train_time_s: float = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QKEModel":
        """Train the QSVC on labelled feature vectors.

        Builds the quantum kernel, computes the N×N training kernel matrix
        (each element = fidelity of two quantum states), then solves the SVM
        dual problem with the resulting Gram matrix.

        Args:
            X: Feature matrix of shape ``(N, 8)`` (StandardScaler-normalized).
            y: Integer labels of shape ``(N,)``; 0 = no planet, 1 = transit.

        Returns:
            self
        """
        from qiskit_machine_learning.algorithms import QSVC
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        self._qsvc = QSVC(
            quantum_kernel=kernel,
            C=self.C,
            class_weight=self.class_weight,
        )

        logger.info(
            "QKE: fitting QSVC on N=%d samples  (C=%.3f, class_weight=%s)",
            len(y), self.C, self.class_weight,
        )
        t0 = time.perf_counter()
        self._qsvc.fit(X, y)
        self.train_time_s = time.perf_counter() - t0
        logger.info("QKE: training complete in %.1f s", self.train_time_s)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Args:
            X: Feature matrix of shape ``(M, 8)``.

        Returns:
            Integer label array of shape ``(M,)``.
        """
        self._check_fitted()
        return self._qsvc.predict(X)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Return the SVM signed margin distance from the decision hyperplane.

        Values > 0 indicate class 1 (transit); values < 0 indicate class 0.
        These scores are monotonically related to the classifier's confidence
        and are suitable as input to ``sklearn.metrics.roc_auc_score``.

        Args:
            X: Feature matrix of shape ``(M, 8)``.

        Returns:
            Float array of shape ``(M,)``.
        """
        self._check_fitted()
        return self._qsvc.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on ``(X, y)``."""
        self._check_fitted()
        return float(self._qsvc.score(X, y))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib.

        Saves the complete ``QKEModel`` object including the fitted QSVC and
        its support vectors.  The quantum kernel is reconstructed at load time
        from ``build_qke_kernel()`` — no circuit parameters need to be stored
        separately.

        Args:
            path: Destination ``.joblib`` file path.
        """
        import joblib
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("QKEModel saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "QKEModel":
        """Load a serialised QKEModel.

        Args:
            path: Path to a ``.joblib`` file written by :meth:`save`.

        Returns:
            Fitted ``QKEModel`` ready for ``predict`` / ``score``.
        """
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected QKEModel, found {type(obj)}")
        logger.info("QKEModel loaded ← %s", path)
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._qsvc is None:
            raise RuntimeError("QKEModel is not fitted. Call fit() first.")


# ── VQC ───────────────────────────────────────────────────────────────────────

class VQCModel:
    """Variational Quantum Classifier with loss tracking.

    Uses ZZFeatureMap (reps=2) + RealAmplitudes (reps=3) with COBYLA
    optimizer.  A callback records the cross-entropy loss at every optimizer
    function evaluation, enabling convergence analysis.

    Training complexity: ``max_iter`` forward passes, each requiring
    ``N`` circuit evaluations (statevector).  For N=60 and max_iter=300,
    expect 5–30 min on a laptop CPU.  Use ``max_iter=50`` for smoke tests.

    Attributes:
        max_iter: COBYLA function-evaluation budget.
        initial_point: Starting parameter values ``(32,)``.  If ``None``,
            random uniform in ``[0, 2π]`` is used.  After :meth:`load`,
            this contains the saved optimal weights for warm-starting.
        loss_history: Cross-entropy loss at each optimizer callback.
        train_time_s: Wall-clock training time in seconds.
    """

    def __init__(self, max_iter: int = 300, initial_point: np.ndarray = None):
        self.max_iter = max_iter
        self.initial_point = initial_point
        self._vqc = None
        self.loss_history: list = []
        self.train_time_s: float = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCModel":
        """Train the VQC on labelled feature vectors.

        Each COBYLA iteration evaluates the cross-entropy loss over all
        training samples using the statevector simulator.  The loss and
        current weights are recorded via a callback after every evaluation.

        Args:
            X: Feature matrix of shape ``(N, 8)`` (StandardScaler-normalized).
            y: Integer labels of shape ``(N,)``; 0 = no planet, 1 = transit.

        Returns:
            self
        """
        from src.quantum.feature_map import build_vqc

        self.loss_history = []

        def _callback(weights, obj_func_eval: float) -> None:
            self.loss_history.append(float(obj_func_eval))
            n = len(self.loss_history)
            if n % 50 == 0 or n == 1:
                logger.info(
                    "VQC iter %4d / %d  loss = %.6f",
                    n, self.max_iter, obj_func_eval,
                )

        self._vqc = build_vqc(
            max_iter=self.max_iter,
            initial_point=self.initial_point,
            callback=_callback,
        )

        logger.info(
            "VQC: fitting on N=%d samples, max_iter=%d", len(y), self.max_iter
        )
        t0 = time.perf_counter()
        self._vqc.fit(X, y)
        self.train_time_s = time.perf_counter() - t0

        final_loss = self.loss_history[-1] if self.loss_history else float("nan")
        logger.info(
            "VQC: done — %d evals, final loss=%.6f, time=%.1f s",
            len(self.loss_history), final_loss, self.train_time_s,
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Args:
            X: Feature matrix of shape ``(M, 8)``.

        Returns:
            Integer label array of shape ``(M,)``.
        """
        self._check_fitted()
        return self._vqc.predict(X)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Return a real-valued classifier confidence score for class 1.

        Accesses the underlying ``SamplerQNN`` forward pass to obtain raw
        output probabilities, which are more informative than hard class
        labels for ROC-AUC computation.  Falls back to the hard prediction
        (0 or 1) if the QNN is not accessible.

        Args:
            X: Feature matrix of shape ``(M, 8)``.

        Returns:
            Float array of shape ``(M,)``; higher = more likely class 1.
        """
        self._check_fitted()
        try:
            nn = self._vqc._neural_network
            raw = nn.forward(X, self._vqc.weights)
            # SamplerQNN output shape: (M, n_classes) or (M, 1)
            if raw.ndim == 2 and raw.shape[1] >= 2:
                return raw[:, 1].astype(float)
            return raw.ravel().astype(float)
        except AttributeError:
            logger.warning("Cannot access VQC neural network; returning hard predictions.")
            return self._vqc.predict(X).astype(float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy on ``(X, y)``."""
        self._check_fitted()
        return float(self._vqc.score(X, y))

    @property
    def weights(self) -> np.ndarray:
        """Optimal VQC parameter vector after training (shape ``(32,)``)."""
        self._check_fitted()
        return np.asarray(self._vqc.weights)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the optimal weights and loss history to disk.

        Creates two files:
            ``<path>.weights.npy``   – optimal parameter array, shape ``(32,)``
            ``<path>.meta.json``     – loss_history, train_time_s, max_iter

        On load, the saved weights become the ``initial_point`` of a new
        VQCModel.  Call :meth:`fit` with the original training data to do a
        warm-start (COBYLA will begin from the saved optimum rather than a
        random point).

        Args:
            path: Base path without extension (e.g. ``"models/vqc"``).
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path + ".weights.npy", self.weights)
        meta = {
            "max_iter": self.max_iter,
            "train_time_s": self.train_time_s,
            "loss_history": self.loss_history,
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("VQCModel weights saved → %s.weights.npy", path)

    @classmethod
    def load(cls, path: str) -> "VQCModel":
        """Load a VQCModel from weights saved by :meth:`save`.

        The returned object is **unfitted** (no internal VQC circuit).
        To make predictions, call :meth:`fit` on the training data — COBYLA
        will warm-start from the saved optimal weights.

        Args:
            path: Base path used in the corresponding :meth:`save` call.

        Returns:
            ``VQCModel`` with ``initial_point``, ``loss_history``, and
            ``train_time_s`` restored.
        """
        weights = np.load(path + ".weights.npy")
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        obj = cls(max_iter=meta["max_iter"], initial_point=weights)
        obj.loss_history = meta["loss_history"]
        obj.train_time_s = meta["train_time_s"]
        logger.info("VQCModel weights loaded ← %s.weights.npy", path)
        return obj

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def convergence_summary(self) -> dict:
        """Return a summary of COBYLA convergence diagnostics.

        Returns:
            dict with keys: n_evals, initial_loss, final_loss, loss_reduction,
            converged (True if loss dropped by > 5 % from initial).
        """
        if not self.loss_history:
            return {}
        initial = self.loss_history[0]
        final = self.loss_history[-1]
        reduction = (initial - final) / (abs(initial) + 1e-12)
        return {
            "n_evals": len(self.loss_history),
            "initial_loss": initial,
            "final_loss": final,
            "loss_reduction": reduction,
            "converged": reduction > 0.05,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._vqc is None:
            raise RuntimeError(
                "VQCModel is not fitted. Call fit(X_train, y_train) first. "
                "After load(), fit() warm-starts from the saved weights."
            )
