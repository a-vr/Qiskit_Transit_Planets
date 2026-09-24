"""Benchmarking utilities: metrics, comparison table, and plots.

Public API
----------
evaluate_model(model, X_test, y_test, name) → dict
    Compute accuracy, precision, recall, F1, ROC-AUC, and training time
    for a single fitted model.

compare_all(named_models, X_test, y_test) → pd.DataFrame
    Evaluate every model in *named_models* and return a tidy DataFrame.

print_table(df) → None
    Print a human-readable comparison table to stdout.

plot_roc_curves(named_models, X_test, y_test, save_path) → Figure
    One ROC curve per model, with AUC in the legend.

plot_loss_curve(vqc_model, save_path) → Figure
    VQC COBYLA convergence plot (loss vs. function evaluation).

plot_feature_importance(X, y, feature_names, save_path) → Figure
    Point-biserial correlation of each feature with the binary label.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Colour palette — distinct, accessible, works in both light and dark themes
_PALETTE = {
    "QKE (QSVC)":       "#4C72B0",   # blue
    "VQC":              "#DD8452",   # orange
    "SVM (RBF)":        "#55A868",   # green
    "Logistic Reg.":    "#C44E52",   # red
}
_FALLBACK_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   name: str) -> dict:
    """Compute classification metrics for a single fitted model.

    Args:
        model: Any fitted model exposing ``predict`` and ``decision_score``.
        X_test: Feature matrix ``(M, 8)``.
        y_test: True labels ``(M,)``.
        name: Display name used as the row key in comparison tables.

    Returns:
        dict with keys: model, accuracy, precision, recall, f1, roc_auc,
        train_time_s.  ``roc_auc`` is ``None`` if only one class is present
        in *y_test*.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)
    scores = model.decision_score(X_test)

    roc_auc = None
    if len(np.unique(y_test)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_test, scores))
        except Exception as exc:
            logger.warning("ROC-AUC computation failed for %s: %s", name, exc)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "train_time_s": getattr(model, "train_time_s", None),
    }


def compare_all(named_models: dict, X_test: np.ndarray,
                y_test: np.ndarray) -> "pd.DataFrame":
    """Evaluate every model and return a tidy comparison DataFrame.

    Args:
        named_models: ``{display_name: fitted_model}`` mapping.
        X_test: Feature matrix ``(M, 8)``.
        y_test: True labels ``(M,)``.

    Returns:
        ``pd.DataFrame`` indexed by model name, columns: accuracy, precision,
        recall, f1, roc_auc, train_time_s.
    """
    import pandas as pd

    rows = [
        evaluate_model(model, X_test, y_test, name)
        for name, model in named_models.items()
    ]
    df = pd.DataFrame(rows).set_index("model")
    return df


def print_table(df: "pd.DataFrame") -> None:
    """Print a formatted comparison table to stdout.

    Args:
        df: DataFrame returned by :func:`compare_all`.
    """
    fmt = {
        "accuracy":     "{:.4f}",
        "precision":    "{:.4f}",
        "recall":       "{:.4f}",
        "f1":           "{:.4f}",
        "roc_auc":      "{:.4f}",
        "train_time_s": "{:.2f} s",
    }
    col_widths = {
        "accuracy": 10, "precision": 11, "recall": 8,
        "f1": 8, "roc_auc": 10, "train_time_s": 16,
    }
    header = f"{'Model':<22}"
    for col, w in col_widths.items():
        header += f"  {col:>{w}}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for model_name, row in df.iterrows():
        line = f"{model_name:<22}"
        for col, w in col_widths.items():
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cell = "    N/A"
            else:
                cell = fmt[col].format(val)
            line += f"  {cell:>{w}}"
        print(line)
    print(sep)
    print()


def save_table_csv(df: "pd.DataFrame", path: str) -> None:
    """Save the comparison DataFrame as a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, float_format="%.6f")
    logger.info("Comparison table saved → %s", path)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _ensure_results_dir(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def plot_roc_curves(
    named_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "results/roc_curves.png",
) -> "Figure":
    """Plot ROC curves for all models on a single axes.

    Args:
        named_models: ``{display_name: fitted_model}`` mapping.
        X_test: Feature matrix ``(M, 8)``.
        y_test: True labels ``(M,)``.
        save_path: Where to save the PNG (also returned as a Figure).

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--",
            linewidth=1, label="Random (AUC = 0.50)")

    for idx, (name, model) in enumerate(named_models.items()):
        try:
            scores = model.decision_score(X_test)
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            color = _PALETTE.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name}  (AUC = {roc_auc:.3f})")
        except Exception as exc:
            logger.warning("ROC curve failed for %s: %s", name, exc)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Transit Planet Classifier Comparison", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _ensure_results_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("ROC curves saved → %s", save_path)
    return fig


def plot_loss_curve(
    vqc_model,
    save_path: str = "results/vqc_loss_curve.png",
) -> "Figure":
    """Plot VQC COBYLA cross-entropy loss vs. function evaluation.

    Args:
        vqc_model: Fitted ``VQCModel`` with a non-empty ``loss_history``.
        save_path: Where to save the PNG.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt

    loss = vqc_model.loss_history
    if not loss:
        raise ValueError("vqc_model.loss_history is empty — model may not be fitted.")

    iters = list(range(1, len(loss) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, loss, color=_PALETTE["VQC"], linewidth=1.5, alpha=0.9)

    # Smoothed trend (running average over window of 10 %)
    window = max(1, len(loss) // 10)
    smoothed = np.convolve(loss, np.ones(window) / window, mode="valid")
    smooth_x = list(range(window, len(loss) + 1))
    ax.plot(smooth_x, smoothed, color="#333333", linewidth=2,
            linestyle="--", label=f"Running avg (window={window})")

    ax.set_xlabel("COBYLA Function Evaluation", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("VQC Training Convergence", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate final loss
    ax.annotate(
        f"Final: {loss[-1]:.4f}",
        xy=(iters[-1], loss[-1]),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#666666"),
    )

    fig.tight_layout()
    _ensure_results_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("VQC loss curve saved → %s", save_path)
    return fig


def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    save_path: str = "results/feature_importance.png",
) -> "Figure":
    """Plot point-biserial correlation of each feature with the binary label.

    A positive correlation means higher feature values associate with transits.
    This gives intuition about which BLS/statistical features are most
    discriminative between transit and non-transit stars.

    Args:
        X: Feature matrix ``(N, 8)`` (StandardScaler-normalized).
        y: Binary labels ``(N,)``.
        feature_names: List of 8 feature name strings.
        save_path: Where to save the PNG.

    Returns:
        ``matplotlib.figure.Figure``
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr

    correlations = []
    for i in range(X.shape[1]):
        corr, _ = pointbiserialr(y, X[:, i])
        correlations.append(corr)

    colors = [_PALETTE["QKE (QSVC)"] if c >= 0 else _PALETTE["Logistic Reg."]
              for c in correlations]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(feature_names, correlations, color=colors, edgecolor="#FFFFFF",
                   linewidth=0.5, height=0.6)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Point-Biserial Correlation with Transit Label", fontsize=11)
    ax.set_title("Feature–Label Correlation (positive = more transits)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, correlations):
        xpos = val + 0.01 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    fig.tight_layout()
    _ensure_results_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Feature importance plot saved → %s", save_path)
    return fig


def plot_all(
    named_models: dict,
    vqc_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    feature_names: list,
    out_dir: str = "results",
) -> None:
    """Convenience wrapper: generate and save all benchmark plots.

    Args:
        named_models: ``{name: model}`` dict for ROC curves.
        vqc_model: Fitted ``VQCModel`` for loss curve.
        X_test, y_test: Test split for ROC curves.
        X_full, y_full: Full dataset for feature importance.
        feature_names: Feature name list from ``src.data.features``.
        out_dir: Directory prefix for all saved plots.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for file output

    plot_roc_curves(named_models, X_test, y_test,
                    save_path=os.path.join(out_dir, "roc_curves.png"))
    plot_loss_curve(vqc_model,
                    save_path=os.path.join(out_dir, "vqc_loss_curve.png"))
    plot_feature_importance(X_full, y_full, feature_names,
                            save_path=os.path.join(out_dir, "feature_importance.png"))
