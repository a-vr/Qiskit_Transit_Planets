"""Classical ML baselines for the transit classifier benchmark.

Two models are provided, each exposing the same interface as
``QKEModel`` and ``VQCModel`` so they can be swapped in transparently:

SVMBaseline
    RBF-kernel SVC — the direct classical analogue of the quantum kernel
    method.  Using the same ``class_weight`` and regularisation C as QKE
    isolates the effect of the quantum kernel from SVM hyperparameters.

LogisticBaseline
    L2-regularised logistic regression — a linear baseline that shows how
    much of the performance gain comes from the non-linear kernel (quantum
    or RBF) vs. simple linear separability.

Both models expose:
    fit / predict / decision_score / score / save / load / train_time_s
"""

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


class SVMBaseline:
    """RBF-kernel Support Vector Classifier.

    Classical analogue of QKEModel.  Using identical ``C`` and
    ``class_weight`` values makes the comparison fair: any performance
    difference is attributable to the kernel, not hyperparameters.

    Attributes:
        C: Regularisation parameter (default 1.0).
        gamma: RBF bandwidth (``'scale'`` = 1 / (n_features × X.var())).
        class_weight: ``'balanced'`` adjusts for class imbalance.
        train_time_s: Wall-clock training time in seconds.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: str = "balanced",
    ):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self._svc = None
        self.train_time_s: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMBaseline":
        """Fit the RBF-SVC on labelled feature vectors.

        Args:
            X: Feature matrix ``(N, 8)``.
            y: Integer labels ``(N,)``; 0 = no planet, 1 = transit.

        Returns:
            self
        """
        from sklearn.svm import SVC

        self._svc = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=42,
        )
        logger.info("SVM: fitting RBF-SVC on N=%d samples", len(y))
        t0 = time.perf_counter()
        self._svc.fit(X, y)
        self.train_time_s = time.perf_counter() - t0
        logger.info("SVM: done in %.3f s", self.train_time_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*."""
        self._check_fitted()
        return self._svc.predict(X)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Return signed SVM margin distances (higher → more likely class 1)."""
        self._check_fitted()
        return self._svc.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        self._check_fitted()
        return float(self._svc.score(X, y))

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* via joblib."""
        import joblib
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("SVMBaseline saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "SVMBaseline":
        """Load a serialised SVMBaseline."""
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SVMBaseline, found {type(obj)}")
        return obj

    def _check_fitted(self) -> None:
        if self._svc is None:
            raise RuntimeError("SVMBaseline is not fitted. Call fit() first.")


class LogisticBaseline:
    """L2-regularised Logistic Regression.

    Linear baseline that tests whether the 8-feature representation is
    linearly separable.  If logistic regression approaches SVM/QKE
    performance, the quantum kernel provides limited benefit over a
    simple linear classifier.

    Attributes:
        C: Inverse regularisation strength (default 1.0).
        class_weight: ``'balanced'`` adjusts for class imbalance.
        max_iter: Solver iteration limit (default 1000).
        train_time_s: Wall-clock training time in seconds.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self._lr = None
        self.train_time_s: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """Fit logistic regression on labelled feature vectors.

        Args:
            X: Feature matrix ``(N, 8)``.
            y: Integer labels ``(N,)``; 0 = no planet, 1 = transit.

        Returns:
            self
        """
        from sklearn.linear_model import LogisticRegression

        self._lr = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=42,
            solver="lbfgs",
        )
        logger.info("LogReg: fitting on N=%d samples", len(y))
        t0 = time.perf_counter()
        self._lr.fit(X, y)
        self.train_time_s = time.perf_counter() - t0
        logger.info("LogReg: done in %.3f s", self.train_time_s)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*."""
        self._check_fitted()
        return self._lr.predict(X)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Return P(class 1) probabilities (higher → more likely transit)."""
        self._check_fitted()
        return self._lr.predict_proba(X)[:, 1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        self._check_fitted()
        return float(self._lr.score(X, y))

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* via joblib."""
        import joblib
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)
        logger.info("LogisticBaseline saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "LogisticBaseline":
        """Load a serialised LogisticBaseline."""
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LogisticBaseline, found {type(obj)}")
        return obj

    def _check_fitted(self) -> None:
        if self._lr is None:
            raise RuntimeError("LogisticBaseline is not fitted. Call fit() first.")
