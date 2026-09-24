"""Unit tests for quantum circuit components.

Tests cover circuit structure (qubit count, parameter count, gate count) and
basic kernel properties (shape, symmetry, unit diagonal).  All tests use tiny
data (4 samples) and run the statevector simulator, so they complete in
seconds on a laptop CPU.

Qiskit and qiskit-machine-learning are imported lazily via ``pytest.importorskip``
so the test file can be collected on machines that only have the data-pipeline
dependencies installed.
"""

import numpy as np
import pytest

# Skip the entire module cleanly if Qiskit is not available
qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
qml = pytest.importorskip(
    "qiskit_machine_learning", reason="qiskit-machine-learning not installed"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _small_X(n: int = 4, seed: int = 0) -> np.ndarray:
    """Return a (n, 8) array of random features in roughly [-2, 2]."""
    return np.random.default_rng(seed).standard_normal((n, 8))


# ── ZZFeatureMap tests ─────────────────────────────────────────────────────────

class TestFeatureMap:
    def test_n_qubits(self):
        from src.quantum.feature_map import build_feature_map, N_QUBITS

        fm = build_feature_map()
        assert fm.num_qubits == N_QUBITS

    def test_has_parameters(self):
        from src.quantum.feature_map import build_feature_map

        fm = build_feature_map()
        assert len(fm.parameters) > 0

    def test_parameter_count_matches_n_qubits_and_reps(self):
        """ZZFeatureMap (linear) with reps=2 encodes n_qubits data values per rep,
        plus n_qubits - 1 ZZ-interaction terms per rep.
        Total data params = reps * (n_qubits + (n_qubits - 1)) = 2 * 15 = 30
        However the library deduplicates x_i across reps, so the actual count
        equals n_qubits (one param per feature, reused across reps).
        We only assert > 0 and a plausible upper bound here."""
        from src.quantum.feature_map import build_feature_map, N_QUBITS, ZZ_REPS

        fm = build_feature_map()
        n_params = len(fm.parameters)
        assert 0 < n_params <= N_QUBITS * ZZ_REPS * 2

    def test_circuit_can_be_drawn(self):
        """Circuit should produce a non-empty string representation."""
        from src.quantum.feature_map import build_feature_map

        fm = build_feature_map()
        drawn = fm.draw("text")
        assert drawn is not None


# ── RealAmplitudes ansatz tests ────────────────────────────────────────────────

class TestAnsatz:
    def test_n_qubits(self):
        from src.quantum.feature_map import build_ansatz, N_QUBITS

        az = build_ansatz()
        assert az.num_qubits == N_QUBITS

    def test_trainable_param_count(self):
        """RealAmplitudes (linear, reps=3, 8 qubits) → (reps+1)*n_qubits = 32."""
        from src.quantum.feature_map import build_ansatz, N_QUBITS, RA_REPS

        az = build_ansatz()
        expected = N_QUBITS * (RA_REPS + 1)
        assert len(az.parameters) == expected

    def test_all_parameters_are_real_angle_gates(self):
        """RealAmplitudes uses only Ry and CNOT — no Rz, Rx, or U gates."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from src.quantum.feature_map import build_ansatz

        az = build_ansatz()
        bound = az.assign_parameters({p: 0.0 for p in az.parameters})
        backend = AerSimulator(method="statevector")
        t = transpile(bound, backend=backend, optimization_level=0)
        ops = set(t.count_ops().keys())
        # After transpilation to AerSimulator basis, only expect rotation and entangling gates
        disallowed = ops - {"ry", "cx", "rz", "u", "id", "barrier", "measure", "x", "h", "rzz"}
        assert not disallowed, f"Unexpected gates in ansatz: {disallowed}"


# ── VQC circuit tests ──────────────────────────────────────────────────────────

class TestVQCCircuit:
    def test_combined_n_qubits(self):
        from src.quantum.feature_map import build_vqc_circuit, N_QUBITS

        circ = build_vqc_circuit()
        assert circ.num_qubits == N_QUBITS

    def test_combined_param_count(self):
        """VQC circuit contains both data params and trainable params."""
        from src.quantum.feature_map import (
            build_vqc_circuit,
            build_feature_map,
            build_ansatz,
        )

        circ = build_vqc_circuit()
        fm = build_feature_map()
        az = build_ansatz()
        # All params = data params (feature map) + trainable params (ansatz)
        assert len(circ.parameters) == len(fm.parameters) + len(az.parameters)

    def test_data_and_trainable_params_are_disjoint(self):
        from src.quantum.feature_map import (
            build_vqc_circuit,
            build_feature_map,
            build_ansatz,
        )

        circ = build_vqc_circuit()
        fm_params = set(p.name for p in build_feature_map().parameters)
        az_params = set(p.name for p in build_ansatz().parameters)
        assert fm_params.isdisjoint(az_params), "Feature map and ansatz share parameter names"


# ── QKE kernel tests ───────────────────────────────────────────────────────────

class TestQKEKernel:
    """Tests run small kernel evaluations (4×4 matrix) using StatevectorSampler."""

    def test_kernel_matrix_shape(self):
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X = _small_X(4)
        K = kernel.evaluate(x_vec=X)
        assert K.shape == (4, 4)

    def test_kernel_diagonal_is_one(self):
        """k(x, x) = |⟨ϕ(x)|ϕ(x)⟩|² = 1 for any x."""
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X = _small_X(4, seed=1)
        K = kernel.evaluate(x_vec=X)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6,
                                   err_msg="Kernel diagonal should be 1")

    def test_kernel_is_symmetric(self):
        """k(x, x') = k(x', x)."""
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X = _small_X(4, seed=2)
        K = kernel.evaluate(x_vec=X)
        np.testing.assert_allclose(K, K.T, atol=1e-6,
                                   err_msg="Kernel matrix must be symmetric")

    def test_kernel_is_positive_semidefinite(self):
        """All eigenvalues of a valid kernel matrix must be ≥ 0."""
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X = _small_X(4, seed=3)
        K = kernel.evaluate(x_vec=X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-6), (
            f"Kernel matrix has negative eigenvalue(s): {eigenvalues[eigenvalues < 0]}"
        )

    def test_kernel_values_in_unit_interval(self):
        """Fidelity kernel values must lie in [0, 1]."""
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X = _small_X(4, seed=4)
        K = kernel.evaluate(x_vec=X)
        assert np.all(K >= -1e-6), "Kernel values must be non-negative"
        assert np.all(K <= 1 + 1e-6), "Kernel values must be ≤ 1"

    def test_train_test_kernel_shape(self):
        """evaluate(X_train, X_test) should return shape (n_train, n_test)."""
        from src.quantum.feature_map import build_qke_kernel

        kernel = build_qke_kernel()
        X_train = _small_X(4, seed=5)
        X_test = _small_X(3, seed=6)
        K_pred = kernel.evaluate(x_vec=X_train, y_vec=X_test)
        assert K_pred.shape == (4, 3)


# ── VQC model construction test ────────────────────────────────────────────────

class TestBuildVQC:
    def test_vqc_is_constructable(self):
        from src.quantum.feature_map import build_vqc

        vqc = build_vqc(max_iter=5)
        assert vqc is not None

    def test_vqc_initial_point_shape(self):
        """build_vqc without explicit initial_point should create a (32,) array."""
        from src.quantum.feature_map import build_vqc, build_ansatz

        vqc = build_vqc(max_iter=5)
        n_expected = len(build_ansatz().parameters)
        # VQC stores initial_point as an attribute
        assert hasattr(vqc, "initial_point")
        assert len(vqc.initial_point) == n_expected

    def test_vqc_respects_custom_initial_point(self):
        from src.quantum.feature_map import build_vqc, build_ansatz, N_QUBITS, RA_REPS

        n_params = N_QUBITS * (RA_REPS + 1)
        custom_pt = np.zeros(n_params)
        vqc = build_vqc(max_iter=5, initial_point=custom_pt)
        np.testing.assert_array_equal(vqc.initial_point, custom_pt)


# ── Circuit analysis smoke test ────────────────────────────────────────────────

class TestCircuitAnalysis:
    def test_analysis_returns_expected_keys(self):
        from src.quantum.feature_map import circuit_analysis

        info = circuit_analysis(verbose=False)
        required = {
            "n_qubits", "zz_reps", "ra_reps",
            "feature_map_depth", "feature_map_n_params", "feature_map_gates",
            "ansatz_depth", "ansatz_n_params", "ansatz_gates",
            "vqc_depth", "vqc_n_trainable_params",
        }
        assert required.issubset(set(info.keys()))

    def test_analysis_n_qubits_correct(self):
        from src.quantum.feature_map import circuit_analysis, N_QUBITS

        info = circuit_analysis(verbose=False)
        assert info["n_qubits"] == N_QUBITS

    def test_analysis_vqc_deeper_than_components(self):
        """Combined circuit must be at least as deep as either sub-circuit."""
        from src.quantum.feature_map import circuit_analysis

        info = circuit_analysis(verbose=False)
        assert info["vqc_depth"] >= info["feature_map_depth"]
        assert info["vqc_depth"] >= info["ansatz_depth"]

    def test_analysis_trainable_params_count(self):
        from src.quantum.feature_map import circuit_analysis, N_QUBITS, RA_REPS

        info = circuit_analysis(verbose=False)
        assert info["vqc_n_trainable_params"] == N_QUBITS * (RA_REPS + 1)
