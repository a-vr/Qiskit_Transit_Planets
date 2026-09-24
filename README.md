# Qiskit Transit Planets

Quantum machine-learning classifier for detecting exoplanet transits in Kepler light curves.

The project trains and compares four classifiers — a Quantum Kernel Estimator (QKE), a
Variational Quantum Classifier (VQC), a classical RBF-SVM, and Logistic Regression — on
8 BLS-derived features extracted from Kepler photometry.

---

## Table of contents

1. [Background](#background)
2. [Architecture](#architecture)
3. [Project layout](#project-layout)
4. [Setup](#setup)
5. [Quickstart](#quickstart)
6. [Running tests](#running-tests)
7. [Benchmark results](#benchmark-results)
8. [Feature descriptions](#feature-descriptions)
9. [Model details](#model-details)

---

## Background

The Kepler space telescope recorded brightness time-series (light curves) for ~200 000 stars.
When a planet crosses its host star the flux dips slightly and periodically — a transit signal.
This project compresses each multi-thousand-point light curve into 8 scalar features and feeds
them into both classical and quantum classifiers to test whether a quantum kernel or variational
circuit offers any advantage over a classical SVM on a real astronomical dataset.

---

## Architecture

```
Kepler FITS / CSV
        │
        ▼
src/data/preprocess.py    remove NaNs → σ-clip outliers → unit-median normalise → Savitzky–Golay flatten
        │
        ▼
src/data/features.py      Box Least Squares → 8 features → StandardScaler → features.npy / labels.npy
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
src/quantum/model.py                            src/classical/baseline.py
  QKEModel  (QSVC + FidelityQuantumKernel)        SVMBaseline  (RBF-SVC)
  VQCModel  (ZZFeatureMap + RealAmplitudes)        LogisticBaseline  (L2 LogReg)
        │                                                  │
        └──────────────┬───────────────────────────────────┘
                       ▼
          src/benchmarking/compare.py
            accuracy / precision / recall / F1 / ROC-AUC / train time
            ROC curves · VQC loss curve · feature-importance bar chart
```

**Quantum circuits (8 qubits)**

| Component | Details |
|-----------|---------|
| Feature map | `ZZFeatureMap` — 8 qubits, `reps=2`, linear entanglement |
| Ansatz | `RealAmplitudes` — 8 qubits, `reps=3`, linear entanglement → **32 trainable parameters** |
| QKE kernel | `FidelityQuantumKernel` via `ComputeUncompute` + `StatevectorSampler` |
| VQC optimiser | COBYLA (gradient-free), up to 300 evaluations |
| Simulator | `AerSimulator` statevector (exact, noiseless) |

---

## Project layout

```
Qiskit_Transit_Planets/
├── data/
│   ├── raw/
│   │   ├── known/          # light curves of confirmed transit hosts
│   │   └── unknown/        # non-host light curves
│   └── processed/
│       ├── features.npy    # (N, 8) float64
│       ├── labels.npy      # (N,) int  0=no-planet  1=transit
│       └── scaler.joblib   # fitted StandardScaler
├── models/                 # saved model artefacts
├── results/                # plots and comparison.csv
├── scripts/
│   ├── build_dataset.py    # download light curves and extract features
│   └── run_benchmark.py    # train all four models and compare
├── src/
│   ├── data/
│   │   ├── download.py     # lightkurve-based Kepler downloader
│   │   ├── preprocess.py   # detrending and normalisation
│   │   └── features.py     # BLS feature extraction and dataset I/O
│   ├── quantum/
│   │   ├── feature_map.py  # ZZFeatureMap, RealAmplitudes, kernel factories
│   │   └── model.py        # QKEModel, VQCModel
│   ├── classical/
│   │   └── baseline.py     # SVMBaseline, LogisticBaseline
│   └── benchmarking/
│       └── compare.py      # metrics, tables, and plots
├── tests/                  # pytest test suite (no internet required)
├── notebooks/              # exploratory Jupyter notebooks
├── requirements.txt
└── pyproject.toml
```

---

## Setup

**Requirements:** Python ≥ 3.9, pip.

```bash
git clone https://github.com/a-vr/Qiskit_Transit_Planets.git
cd Qiskit_Transit_Planets

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Classical models and all tests run without Qiskit.
Quantum models additionally require `qiskit`, `qiskit-machine-learning`, `qiskit-aer`, and
`qiskit-algorithms` (all listed in `requirements.txt`).

---

## Quickstart

### 1 — Build the dataset

Downloads Kepler light curves for confirmed-planet hosts and non-host stars, extracts 8 BLS
features per star, and writes the feature matrix to `data/processed/`.

```bash
python scripts/build_dataset.py --known 80 --unknown 80
```

This requires internet access and downloads ~160 FITS files (~300 MB).
Expect 5–20 minutes depending on the MAST archive.

### 2 — Run the benchmark

```bash
# All four models (slow — QKE is O(N²), VQC runs 300 COBYLA iterations)
python scripts/run_benchmark.py

# Classical baselines only (completes in < 5 s)
python scripts/run_benchmark.py --skip-quantum

# Fewer VQC iterations for a quick end-to-end test
python scripts/run_benchmark.py --vqc-iter 50
```

Output is written to `results/`:

```
results/
├── comparison.csv
├── roc_curves.png
├── vqc_loss_curve.png
└── feature_importance.png
```

### 3 — Use models directly

```python
import numpy as np
from src.data.features import load_dataset, stratified_split
from src.quantum.model import QKEModel, VQCModel
from src.classical.baseline import SVMBaseline

X, y, scaler = load_dataset("data/processed")
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

# Classical SVM
svm = SVMBaseline(C=1.0, class_weight="balanced").fit(X_train, y_train)
print(f"SVM accuracy: {svm.score(X_test, y_test):.3f}")

# Quantum kernel estimator
qke = QKEModel(C=1.0, class_weight="balanced").fit(X_train, y_train)
print(f"QKE accuracy: {qke.score(X_test, y_test):.3f}")

# Variational quantum classifier
vqc = VQCModel(max_iter=300).fit(X_train, y_train)
print(f"VQC accuracy: {vqc.score(X_test, y_test):.3f}")
print(vqc.convergence_summary())
```

---

## Running tests

The test suite uses synthetic data only — no internet access and no Kepler downloads.

```bash
# Fast run (classical + benchmarking tests; no Qiskit required)
pytest tests/test_benchmarking.py -v

# Full suite (requires Qiskit packages)
pytest tests/ -v --timeout=600

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests are automatically run on every push and pull request to `main` via GitHub Actions
(`.github/workflows/ci.yml`).

---

## Benchmark results

Run `python scripts/run_benchmark.py` after building the dataset to populate this table.
The figures below are illustrative targets based on similar Kepler studies.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|-------|----------|-----------|--------|----|---------|------------|
| QKE (QSVC) | — | — | — | — | — | — |
| VQC | — | — | — | — | — | — |
| SVM (RBF) | — | — | — | — | — | — |
| Logistic Reg. | — | — | — | — | — | — |

*Fill in after running `scripts/run_benchmark.py` and copying `results/comparison.csv`.*

---

## Feature descriptions

Eight features are extracted per star via Box Least Squares (BLS) period search
(`numpy`-based, period grid 0.5–30 days, 500 points):

| Index | Name | Description |
|-------|------|-------------|
| 0 | `bls_peak_power` | Normalised BLS peak power (signal-to-noise proxy) |
| 1 | `transit_depth` | Fractional flux decrease at best-fit period |
| 2 | `transit_duration_frac` | Transit duration as fraction of orbital period |
| 3 | `log_period_days` | log₁₀ of best-fit period in days |
| 4 | `transit_snr` | Transit depth divided by out-of-transit RMS scatter |
| 5 | `rms_scatter` | RMS of the flattened, normalised flux |
| 6 | `flux_skewness` | Skewness of the flux distribution (transits → negative tail) |
| 7 | `flux_kurtosis` | Excess kurtosis of the flux distribution |

All features are standardised (zero mean, unit variance) before being fed to the models.

---

## Model details

### QKEModel

Wraps scikit-learn's `QSVC` with a `FidelityQuantumKernel`.  The kernel computes
k(x, x') = |⟨ϕ(x)|ϕ(x')⟩|² where |ϕ(x)⟩ is the state prepared by the `ZZFeatureMap`.
Training is O(N²) kernel evaluations; use `--skip-quantum` for large datasets.

**Save / load**

```python
model.save("models/qke.joblib")
loaded = QKEModel.load("models/qke.joblib")
```

### VQCModel

Uses Qiskit Machine Learning's `VQC` with a COBYLA optimiser.  A callback records the
cross-entropy loss at every function evaluation.  The model can be saved and warm-started:

```python
model.save("models/vqc")           # writes vqc.weights.npy + vqc.meta.json
warm = VQCModel.load("models/vqc")
warm.fit(X_train, y_train)         # continues from saved weights
```

### SVMBaseline / LogisticBaseline

Direct classical analogues of QKEModel and VQCModel.  Using the same `C` and
`class_weight` values makes the comparison fair: any performance difference is
attributable to the kernel, not hyperparameter tuning.
