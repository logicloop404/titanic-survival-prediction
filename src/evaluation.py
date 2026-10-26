"""
evaluation.py
=============
Compute and visualise classification metrics.

Functions
---------
evaluate_model       – compute accuracy, precision, recall, F1, confusion matrix
plot_confusion_matrix – save a heatmap PNG to outputs/
plot_model_comparison – save a grouped bar chart PNG to outputs/
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts & Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def evaluate_model(y_true, y_pred, model_name: str = "") -> dict:
    """
    Compute binary classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix, report
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, target_names=["Not Survived", "Survived"]),
    }


def plot_confusion_matrix(
    metrics: dict,
    save: bool = True,
) -> pathlib.Path | None:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    metrics : dict returned by evaluate_model
    save    : if True, write PNG to outputs/

    Returns
    -------
    Path to the saved file, or None if save=False.
    """
    cm = metrics["confusion_matrix"]
    name = metrics.get("model_name", "Model")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.lower().replace(" ", "_")
        filepath = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    plt.close(fig)
    return None


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    save: bool = True,
) -> pathlib.Path | None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, and F1 Score
    across all models.

    Parameters
    ----------
    metrics_df : DataFrame with columns [Model, Accuracy, Precision, Recall, F1 Score]
    save       : if True, write PNG to outputs/
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    melted = metrics_df.melt(
        id_vars="Model",
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("Set2", n_colors=len(metrics_to_plot))
    sns.barplot(
        data=melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax,
        edgecolor="white",
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Classification Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(title="Metric", loc="lower right")
    ax.tick_params(axis="x", rotation=15)

    # Annotate bar tops
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    plt.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filepath = OUTPUT_DIR / "model_comparison.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    plt.close(fig)
    return None


def plot_feature_importance(pipeline, feature_names: list[str], save: bool = True):
    """
    Plot feature importances for tree-based models (Random Forest, Decision Tree).
    Skips silently for models that don't expose feature_importances_.
    """
    clf = pipeline.named_steps.get("classifier")
    if not hasattr(clf, "feature_importances_"):
        return None

    # Get encoded feature names from the preprocessor
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        enc_features = preprocessor.get_feature_names_out()
    except Exception:
        enc_features = [f"f{i}" for i in range(len(clf.feature_importances_))]

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]   # top 15

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color=sns.color_palette("Blues_d", len(indices)),
        edgecolor="white",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([enc_features[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importances (Top 15)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filepath = OUTPUT_DIR / "feature_importance.png"
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    plt.close(fig)
    return None
