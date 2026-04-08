"""Train Explainable Boosting Machine (EBM) for danger scoring."""

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from rl_testing.features.feature_extractor import FEATURE_NAMES


def train_ebm(
    results_dir: str = "results",
    seed: int = 42,
) -> ExplainableBoostingClassifier:
    """Train EBM on step-level features with hazardous labels.

    Args:
        results_dir: Base results directory.
        seed: Random seed.

    Returns:
        Trained EBM model.
    """
    print("=" * 60)
    print("Step 3: Training EBM")
    print("=" * 60)

    stats_dir = os.path.join(results_dir, "statistics")
    ebm_dir = os.path.join(results_dir, "ebm_model")
    os.makedirs(ebm_dir, exist_ok=True)

    # Load data
    feature_matrix = np.load(os.path.join(stats_dir, "feature_matrix.npy"))
    hazardous_labels = np.load(os.path.join(stats_dir, "hazardous_labels.npy"))
    episode_ids = np.load(os.path.join(stats_dir, "episode_ids.npy"))
    episode_labels = np.load(os.path.join(stats_dir, "episode_labels.npy"))

    # 80/20 split by episode
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episodes)
    n_train = int(0.8 * len(unique_episodes))
    train_eps = set(unique_episodes[:n_train].tolist())
    val_eps = set(unique_episodes[n_train:].tolist())

    train_mask = np.isin(episode_ids, list(train_eps))
    val_mask = np.isin(episode_ids, list(val_eps))

    X_train = feature_matrix[train_mask]
    y_train = hazardous_labels[train_mask]
    X_val = feature_matrix[val_mask]
    y_val = hazardous_labels[val_mask]

    print(f"  Train: {X_train.shape[0]} steps from {n_train} episodes")
    print(f"  Val: {X_val.shape[0]} steps from {len(unique_episodes) - n_train} episodes")
    print(f"  Train hazardous rate: {y_train.mean():.4f}")
    print(f"  Val hazardous rate: {y_val.mean():.4f}")

    # Train EBM
    print("  Training EBM...")
    ebm = ExplainableBoostingClassifier(
        feature_names=FEATURE_NAMES,
        random_state=seed,
    )
    ebm.fit(X_train, y_train)

    # Step-level evaluation
    val_proba = ebm.predict_proba(X_val)[:, 1]
    step_auroc = roc_auc_score(y_val, val_proba)
    step_auprc = average_precision_score(y_val, val_proba)

    print(f"\n  Step-level AUROC: {step_auroc:.4f}")
    print(f"  Step-level AUPRC: {step_auprc:.4f}")

    # Episode-level evaluation
    val_episode_ids = episode_ids[val_mask]
    val_ep_scores = {}
    val_ep_labels = {}

    for ep_id in val_eps:
        ep_mask = val_episode_ids == ep_id
        if ep_mask.any():
            ep_proba = val_proba[ep_mask]
            val_ep_scores[ep_id] = float(np.max(ep_proba))
            val_ep_labels[ep_id] = int(episode_labels[ep_id])

    ep_scores_arr = np.array([val_ep_scores[k] for k in sorted(val_ep_scores.keys())])
    ep_labels_arr = np.array([val_ep_labels[k] for k in sorted(val_ep_labels.keys())])

    if len(np.unique(ep_labels_arr)) > 1:
        ep_auroc = roc_auc_score(ep_labels_arr, ep_scores_arr)
        ep_auprc = average_precision_score(ep_labels_arr, ep_scores_arr)
    else:
        ep_auroc = float("nan")
        ep_auprc = float("nan")

    print(f"  Episode-level AUROC: {ep_auroc:.4f}")
    print(f"  Episode-level AUPRC: {ep_auprc:.4f}")

    # Feature importance
    importances = {}
    global_explanation = ebm.explain_global()
    for i, name in enumerate(FEATURE_NAMES):
        importances[name] = float(global_explanation.data(i)["scores"].mean())

    # Sort by absolute importance
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Feature importance ranking:")
    for name, imp in sorted_imp:
        print(f"    {name}: {imp:.4f}")

    # Save model
    with open(os.path.join(ebm_dir, "ebm_model.pkl"), "wb") as f:
        pickle.dump(ebm, f)

    # Save feature importance
    with open(os.path.join(ebm_dir, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)

    # Save metrics
    metrics = {
        "step_auroc": step_auroc,
        "step_auprc": step_auprc,
        "episode_auroc": ep_auroc if not np.isnan(ep_auroc) else None,
        "episode_auprc": ep_auprc if not np.isnan(ep_auprc) else None,
    }
    with open(os.path.join(ebm_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot EBM shape functions
    try:
        fig = global_explanation.visualize()
        if hasattr(fig, "savefig"):
            fig.savefig(os.path.join(ebm_dir, "ebm_shape_functions.png"), dpi=300)
        elif hasattr(fig, "write_image"):
            fig.write_image(os.path.join(ebm_dir, "ebm_shape_functions.png"))
        else:
            # Plotly figure - save as HTML
            import plotly
            plotly.io.write_html(fig, os.path.join(ebm_dir, "ebm_shape_functions.html"))
    except Exception as e:
        print(f"  Warning: Could not save EBM shape functions: {e}")

    # Plot ROC and PR curves
    _plot_roc_pr(y_val, val_proba, ebm_dir, prefix="step")
    if len(np.unique(ep_labels_arr)) > 1:
        _plot_roc_pr(ep_labels_arr, ep_scores_arr, ebm_dir, prefix="episode")

    return ebm


def _plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_dir: str,
    prefix: str,
) -> None:
    """Plot and save ROC and PR curves."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{prefix.capitalize()}-level ROC Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_roc.png"), dpi=300)
    plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{prefix.capitalize()}-level PR Curve", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_pr.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    train_ebm()
