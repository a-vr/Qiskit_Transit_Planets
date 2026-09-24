"""End-to-end benchmark: train all four models and compare results.

Usage
-----
    python scripts/run_benchmark.py [--data-dir data/processed] [--out-dir results]

Prerequisites
-------------
Run ``python scripts/build_dataset.py`` first to download light curves and
extract features into ``data/processed/features.npy`` and ``labels.npy``.

What this script does
---------------------
1. Load the pre-built feature matrix from ``data/processed/``.
2. Stratified 70 / 15 / 15 train / val / test split.
3. Train four models on the training split:
       QKE   — Quantum Kernel Estimator (QSVC + FidelityQuantumKernel)
       VQC   — Variational Quantum Classifier (ZZFeatureMap + RealAmplitudes)
       SVM   — Classical RBF-SVC
       LogReg — L2 Logistic Regression
4. Evaluate each model on the held-out test split.
5. Print a formatted comparison table.
6. Save the table as ``results/comparison.csv``.
7. Save ROC curves, VQC loss curve, and feature importance plots to
   ``results/``.
8. Save all fitted models to ``models/``.

Notes
-----
- QKE training is O(N²) kernel evaluations.  Expect 10–120 min for N=60.
- VQC training runs ``--vqc-iter`` COBYLA evaluations.  Expect 30–300 min.
- Classical models train in < 1 s.
"""

import argparse
import logging
import os
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import

from src.data.features import load_dataset, stratified_split, FEATURE_NAMES
from src.quantum.model import QKEModel, VQCModel
from src.classical.baseline import SVMBaseline, LogisticBaseline
from src.benchmarking.compare import compare_all, print_table, save_table_csv, plot_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantum vs classical transit classifiers.")
    parser.add_argument("--data-dir", default="data/processed",
                        help="Directory containing features.npy and labels.npy")
    parser.add_argument("--out-dir", default="results",
                        help="Directory for plots and comparison CSV")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save fitted models")
    parser.add_argument("--vqc-iter", type=int, default=300,
                        help="COBYLA max iterations for VQC (default 300; use 50 for quick test)")
    parser.add_argument("--skip-quantum", action="store_true",
                        help="Skip QKE and VQC (classical baselines only, for quick testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    features_path = os.path.join(args.data_dir, "features.npy")
    if not os.path.exists(features_path):
        print(
            f"ERROR: {features_path} not found.\n"
            "Run 'python scripts/build_dataset.py' first to download and featurise "
            "the Kepler light curves."
        )
        sys.exit(1)

    X, y, scaler = load_dataset(args.data_dir)
    pos, total = int(y.sum()), len(y)
    logger.info("Dataset: %d samples  |  %d transit hosts  |  %d non-hosts",
                total, pos, total - pos)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    logger.info("Split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))

    # ── 3. Train ─────────────────────────────────────────────────────────────
    named_models = {}
    vqc_model = None

    if not args.skip_quantum:
        logger.info("═" * 50)
        logger.info("Training QKE (QSVC + FidelityQuantumKernel)…")
        qke = QKEModel(C=1.0, class_weight="balanced").fit(X_train, y_train)
        qke.save(os.path.join(args.model_dir, "qke.joblib"))
        named_models["QKE (QSVC)"] = qke

        logger.info("═" * 50)
        logger.info("Training VQC (max_iter=%d)…", args.vqc_iter)
        vqc = VQCModel(max_iter=args.vqc_iter).fit(X_train, y_train)
        vqc.save(os.path.join(args.model_dir, "vqc"))
        named_models["VQC"] = vqc
        vqc_model = vqc

        summary = vqc.convergence_summary()
        logger.info(
            "VQC convergence: %d evals  initial=%.4f  final=%.4f  reduction=%.1f%%  converged=%s",
            summary["n_evals"], summary["initial_loss"], summary["final_loss"],
            summary["loss_reduction"] * 100, summary["converged"],
        )

    logger.info("═" * 50)
    logger.info("Training SVM (RBF kernel)…")
    svm = SVMBaseline(C=1.0, class_weight="balanced").fit(X_train, y_train)
    svm.save(os.path.join(args.model_dir, "svm.joblib"))
    named_models["SVM (RBF)"] = svm

    logger.info("Training Logistic Regression…")
    lr = LogisticBaseline(C=1.0, class_weight="balanced").fit(X_train, y_train)
    lr.save(os.path.join(args.model_dir, "logreg.joblib"))
    named_models["Logistic Reg."] = lr

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    logger.info("═" * 50)
    logger.info("Evaluating on held-out test set (N=%d)…", len(y_test))
    results_df = compare_all(named_models, X_test, y_test)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  BENCHMARK RESULTS  (test set)")
    print("═" * 70)
    print_table(results_df)

    csv_path = os.path.join(args.out_dir, "comparison.csv")
    save_table_csv(results_df, csv_path)
    print(f"Table saved → {csv_path}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    logger.info("Generating plots → %s/", args.out_dir)
    plot_roc = True
    plot_loss = vqc_model is not None and len(vqc_model.loss_history) > 0

    from src.benchmarking.compare import (
        plot_roc_curves, plot_loss_curve, plot_feature_importance,
    )
    if plot_roc:
        plot_roc_curves(named_models, X_test, y_test,
                        save_path=os.path.join(args.out_dir, "roc_curves.png"))
    if plot_loss:
        plot_loss_curve(vqc_model,
                        save_path=os.path.join(args.out_dir, "vqc_loss_curve.png"))
    plot_feature_importance(X, y, FEATURE_NAMES,
                            save_path=os.path.join(args.out_dir, "feature_importance.png"))

    print(f"\nPlots saved to {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
