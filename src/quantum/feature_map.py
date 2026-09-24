"""Quantum circuit components for the Kepler transit classifier.

Two models are built and compared against each other and against classical ML:

QKE — Quantum Kernel Estimator
───────────────────────────────
A ZZFeatureMap encodes the 8 classical features into a quantum state.  The
squared overlap between two such states defines a quantum kernel

    k(x, x') = |⟨ϕ(x)|ϕ(x')⟩|²

which is passed to a classical SVC (QSVC = SVC with quantum kernel).  No
parameters are trained; the kernel matrix is computed once and reused.

Why ZZFeatureMap?
    • Encodes real-valued data via Pauli-Z rotations and ZZ two-qubit
      interactions — exactly the class of circuits conjectured to be hard
      to simulate classically on large qubit counts.
    • The Hilbert-space feature dimension is 2^n = 256 for n = 8 qubits,
      far larger than any practical classical feature expansion.
    • With ``entanglement='linear'`` the gate count is O(n) per layer, keeping
      circuit depth tractable for statevector simulation.
    • reps=2 applies the encoding block twice, increasing expressibility while
      keeping transpiled depth below 30 on AerSimulator.

VQC — Variational Quantum Classifier
──────────────────────────────────────
The same ZZFeatureMap is prepended to a trainable RealAmplitudes ansatz.
Parameters θ are optimized end-to-end to minimize cross-entropy loss.

Why RealAmplitudes?
    • Uses only Ry and CNOT gates (all matrix elements real-valued), halving
      the parameter count compared to a full Euler-angle (ZYZ) ansatz.
    • With reps=3 and 8 qubits: (reps+1) × n_qubits = 32 trainable
      parameters — small enough for COBYLA to converge in ≤ 500 evaluations.
    • Known to avoid the worst barren-plateau regimes that affect very deep
      alternating-layered ansätze.
    • Fully compatible with AerSimulator's noiseless statevector mode.

Optimizer choice: COBYLA
    Constrained Optimization BY Linear Approximations.  Gradient-free, so no
    parameter-shift overhead.  Well-suited for ≤ ~100 parameters and moderate
    function evaluations, which is exactly our regime.

Simulator
    AerSimulator in statevector mode via ``StatevectorSampler`` from
    ``qiskit.primitives``.  No shot noise for baseline experiments; noisy
    simulation is a future extension.
"""

import numpy as np

# ── Circuit constants ──────────────────────────────────────────────────────────
N_QUBITS = 8   # one qubit per feature; set in src/data/features.py (N_FEATURES)
ZZ_REPS = 2    # ZZFeatureMap repetitions
RA_REPS = 3    # RealAmplitudes repetitions
# Derived: RealAmplitudes trainable parameters = N_QUBITS × (RA_REPS + 1) = 32


# ── Building blocks ────────────────────────────────────────────────────────────

def build_feature_map():
    """Return the shared ZZFeatureMap used by both QKE and VQC.

    Returns:
        ``qiskit.circuit.library.ZZFeatureMap`` with 8 qubits, 2 reps,
        linear entanglement.
    """
    from qiskit.circuit.library import ZZFeatureMap

    return ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=ZZ_REPS,
        entanglement="linear",
    )


def build_ansatz():
    """Return the RealAmplitudes ansatz used by VQC.

    Returns:
        ``qiskit.circuit.library.RealAmplitudes`` with 8 qubits, 3 reps,
        linear entanglement.  Contains 32 trainable Ry-gate parameters.
    """
    from qiskit.circuit.library import RealAmplitudes

    return RealAmplitudes(
        num_qubits=N_QUBITS,
        reps=RA_REPS,
        entanglement="linear",
    )


def build_vqc_circuit():
    """Return the full VQC circuit (ZZFeatureMap composed with RealAmplitudes).

    The circuit has two disjoint parameter sets:
        - ``feature_map.parameters``: data parameters (bound at inference time)
        - ``ansatz.parameters``: trainable weights (optimized during fit)

    Returns:
        ``qiskit.circuit.QuantumCircuit`` representing feature_map ∘ ansatz.
    """
    feature_map = build_feature_map()
    ansatz = build_ansatz()
    return feature_map.compose(ansatz)


# ── QKE kernel ────────────────────────────────────────────────────────────────

def build_qke_kernel():
    """Build a ``FidelityQuantumKernel`` for QKE / QSVC.

    Kernel function: k(x, x') = |⟨ϕ(x)|ϕ(x')⟩|²

    Implemented via the ``ComputeUncompute`` fidelity estimator: for each
    pair (x, x') a circuit U(x)†·U(x') is run and the probability of
    measuring the all-zeros bitstring gives the kernel value.

    Uses ``StatevectorSampler`` (exact, noiseless).

    Returns:
        ``qiskit_machine_learning.kernels.FidelityQuantumKernel``
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute

    feature_map = build_feature_map()
    fidelity = ComputeUncompute(sampler=StatevectorSampler())
    return FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)


# ── VQC model ─────────────────────────────────────────────────────────────────

def build_vqc(max_iter: int = 300, initial_point: np.ndarray = None, callback=None):
    """Build a ``VQC`` (Variational Quantum Classifier).

    The VQC uses cross-entropy loss with a sigmoid output layer.  At each
    COBYLA iteration the full statevector is computed — exact, no shot noise.

    Args:
        max_iter: Maximum COBYLA function evaluations (default 300).
        initial_point: Optional starting parameter values, shape ``(32,)``.
            If ``None``, uses random initialization from ``[0, 2π]``.
        callback: Optional callable ``(weights, obj_func_eval) → None``
            invoked at each optimizer step.  Must be passed at construction
            time — assignment after construction is not forwarded to the
            optimizer in newer versions of qiskit-machine-learning.

    Returns:
        ``qiskit_machine_learning.algorithms.VQC``
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQC

    feature_map = build_feature_map()
    ansatz = build_ansatz()

    if initial_point is None:
        rng = np.random.default_rng(42)
        initial_point = rng.uniform(0, 2 * np.pi, len(ansatz.parameters))

    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=max_iter),
        sampler=StatevectorSampler(),
        initial_point=initial_point,
        callback=callback,
    )


# ── Circuit analysis ──────────────────────────────────────────────────────────

def circuit_analysis(verbose: bool = True) -> dict:
    """Analyse and optionally print the circuit depth and gate counts.

    Transpiles the ZZFeatureMap, RealAmplitudes ansatz, and combined VQC
    circuit against ``AerSimulator(method='statevector')`` at
    ``optimization_level=1`` to report realistic gate counts.

    Args:
        verbose: If True, print a formatted summary table.

    Returns:
        dict with keys:
            n_qubits, zz_reps, ra_reps,
            feature_map_depth, feature_map_n_params, feature_map_gates,
            ansatz_depth, ansatz_n_params, ansatz_gates,
            vqc_depth, vqc_n_trainable_params.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = AerSimulator(method="statevector")

    feature_map = build_feature_map()
    ansatz = build_ansatz()
    vqc_circuit = build_vqc_circuit()

    def _bind_zeros(circuit):
        """Bind all symbolic parameters to 0 so the circuit can be transpiled."""
        return circuit.assign_parameters(
            {p: 0.0 for p in circuit.parameters}
        )

    fm_t = transpile(_bind_zeros(feature_map), backend=backend, optimization_level=1)
    az_t = transpile(_bind_zeros(ansatz), backend=backend, optimization_level=1)
    vqc_t = transpile(_bind_zeros(vqc_circuit), backend=backend, optimization_level=1)

    info = {
        "n_qubits": N_QUBITS,
        "zz_reps": ZZ_REPS,
        "ra_reps": RA_REPS,
        "feature_map_depth": fm_t.depth(),
        "feature_map_n_params": len(feature_map.parameters),
        "feature_map_gates": dict(fm_t.count_ops()),
        "ansatz_depth": az_t.depth(),
        "ansatz_n_params": len(ansatz.parameters),
        "ansatz_gates": dict(az_t.count_ops()),
        "vqc_depth": vqc_t.depth(),
        "vqc_n_trainable_params": len(ansatz.parameters),
    }

    if verbose:
        _print_analysis(info)

    return info


def _print_analysis(info: dict) -> None:
    sep = "─" * 56
    print(f"\n{'═' * 56}")
    print(f"  CIRCUIT ANALYSIS  (AerSimulator statevector, opt=1)")
    print(f"{'═' * 56}")
    print(f"  Qubits                    : {info['n_qubits']}")
    print()
    print(f"  ZZFeatureMap  (reps={info['zz_reps']})")
    print(f"  {sep}")
    print(f"    transpiled depth        : {info['feature_map_depth']}")
    print(f"    data parameters         : {info['feature_map_n_params']}")
    print(f"    gate counts             : {info['feature_map_gates']}")
    print()
    print(f"  RealAmplitudes  (reps={info['ra_reps']})")
    print(f"  {sep}")
    print(f"    transpiled depth        : {info['ansatz_depth']}")
    print(f"    trainable parameters    : {info['ansatz_n_params']}")
    print(f"    gate counts             : {info['ansatz_gates']}")
    print()
    print(f"  VQC  (ZZFeatureMap ∘ RealAmplitudes)")
    print(f"  {sep}")
    print(f"    transpiled depth        : {info['vqc_depth']}")
    print(f"    trainable parameters    : {info['vqc_n_trainable_params']}")
    print(f"{'═' * 56}\n")
