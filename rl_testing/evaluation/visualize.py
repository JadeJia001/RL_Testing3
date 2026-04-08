"""Visualization: generate all 10 figures for experiment results."""

import json
import os
import pickle
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from rl_testing.features.feature_extractor import FEATURE_NAMES
from rl_testing.evaluation.evaluate import compute_ece, get_bin_index


# Academic style defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
})


def generate_all_figures(results_dir: str = "results") -> None:
    """Generate all 10 figures and save to results/figures/."""
    print("=" * 60)
    print("Step 6: Generating Visualizations")
    print("=" * 60)

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    with open(os.path.join(results_dir, "g0_random", "episodes.pkl"), "rb") as f:
        g0_episodes = pickle.load(f)
    with open(os.path.join(results_dir, "g1_evolutionary", "all_episodes.pkl"), "rb") as f:
        g1_episodes = pickle.load(f)
    with open(os.path.join(results_dir, "g1_evolutionary", "convergence.json"), "r") as f:
        convergence = json.load(f)
    with open(os.path.join(results_dir, "ebm_model", "feature_importance.json"), "r") as f:
        feat_imp = json.load(f)
    with open(os.path.join(results_dir, "statistics", "quantile_boundaries.pkl"), "rb") as f:
        quantile_boundaries = pickle.load(f)

    sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [name for name, _ in sorted_features[:3]]
    top_indices = [FEATURE_NAMES.index(name) for name in top_features]

    g0_scores = np.array([ep["episode_score"] for ep in g0_episodes])
    g0_labels = np.array([ep.get("failure_label", 0) for ep in g0_episodes])
    g1_scores = np.array([ep["episode_score"] for ep in g1_episodes])
    g1_labels = np.array([ep.get("failure_label", 0) for ep in g1_episodes])

    # 1. ROC curve
    _plot_roc(g0_scores, g0_labels, g1_scores, g1_labels, fig_dir)

    # 2. PR curve
    _plot_pr(g0_scores, g0_labels, g1_scores, g1_labels, fig_dir)

    # 3. Calibration diagram
    _plot_calibration(g0_scores, g0_labels, g1_scores, g1_labels, fig_dir)

    # 4. Failure rate comparison
    _plot_failure_rate(g0_labels, g1_labels, fig_dir)

    # 5. Episode score distribution
    _plot_score_distribution(g0_scores, g1_scores, fig_dir)

    # 6. Search convergence
    _plot_convergence(convergence, fig_dir)

    # 7. Feature importance
    _plot_feature_importance(feat_imp, fig_dir)

    # 8. Cumulative failure curve
    _plot_cumulative_failures(g0_labels, g1_labels, fig_dir)

    # 9. Perturbation space
    _plot_perturbation_space(g0_episodes, g1_episodes, fig_dir)

    # 10. Failure coverage heatmap
    _plot_failure_coverage_heatmap(
        g0_episodes, g1_episodes, top_features, top_indices,
        quantile_boundaries, fig_dir,
    )

    print(f"  All 10 figures saved to {fig_dir}/")


def _plot_roc(
    g0_scores: np.ndarray, g0_labels: np.ndarray,
    g1_scores: np.ndarray, g1_labels: np.ndarray,
    fig_dir: str,
) -> None:
    """1. ROC curves for G0 and G1."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for scores, labels, name in [(g0_scores, g0_labels, "G0 Random"),
                                  (g1_scores, g1_labels, "G1 Evolutionary")]:
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: G0 vs G1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=300)
    plt.close(fig)
    print("  1/10 roc_curve.png")


def _plot_pr(
    g0_scores: np.ndarray, g0_labels: np.ndarray,
    g1_scores: np.ndarray, g1_labels: np.ndarray,
    fig_dir: str,
) -> None:
    """2. PR curves for G0 and G1."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for scores, labels, name in [(g0_scores, g0_labels, "G0 Random"),
                                  (g1_scores, g1_labels, "G1 Evolutionary")]:
        if len(np.unique(labels)) > 1:
            prec, rec, _ = precision_recall_curve(labels, scores)
            ap = average_precision_score(labels, scores)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves: G0 vs G1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=300)
    plt.close(fig)
    print("  2/10 pr_curve.png")


def _plot_calibration(
    g0_scores: np.ndarray, g0_labels: np.ndarray,
    g1_scores: np.ndarray, g1_labels: np.ndarray,
    fig_dir: str,
) -> None:
    """3. Calibration / reliability diagram."""
    fig, ax = plt.subplots(figsize=(7, 6))
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for scores, labels, name, color in [
        (g0_scores, g0_labels, "G0 Random", "C0"),
        (g1_scores, g1_labels, "G1 Evolutionary", "C1"),
    ]:
        bin_centers = []
        bin_rates = []
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
            else:
                mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append(scores[mask].mean())
                bin_rates.append(labels[mask].mean())

        if bin_centers:
            ax.plot(bin_centers, bin_rates, "o-", label=name, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Actual Failure Rate")
    ax.set_title("Calibration Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "calibration_diagram.png"), dpi=300)
    plt.close(fig)
    print("  3/10 calibration_diagram.png")


def _plot_failure_rate(
    g0_labels: np.ndarray, g1_labels: np.ndarray, fig_dir: str,
) -> None:
    """4. Failure rate bar chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    rates = [g0_labels.mean(), g1_labels.mean()]
    bars = ax.bar(["G0 Random", "G1 Evolutionary"], rates,
                  color=["C0", "C1"], width=0.5, edgecolor="black")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.3f}", ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Failure Rate")
    ax.set_title("Failure Rate: G0 vs G1")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "failure_rate_comparison.png"), dpi=300)
    plt.close(fig)
    print("  4/10 failure_rate_comparison.png")


def _plot_score_distribution(
    g0_scores: np.ndarray, g1_scores: np.ndarray, fig_dir: str,
) -> None:
    """5. Episode score distribution histograms."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(g0_scores, bins=bins, alpha=0.5, label="G0 Random", color="C0", edgecolor="black")
    ax.hist(g1_scores, bins=bins, alpha=0.5, label="G1 Evolutionary", color="C1", edgecolor="black")
    ax.set_xlabel("Episode Score")
    ax.set_ylabel("Count")
    ax.set_title("Episode Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "episode_score_distribution.png"), dpi=300)
    plt.close(fig)
    print("  5/10 episode_score_distribution.png")


def _plot_convergence(convergence: List[Dict], fig_dir: str) -> None:
    """6. Search convergence curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = [c["generation"] for c in convergence]
    best_scores = [c["best_score"] for c in convergence]

    ax.plot(gens, best_scores, label="Best Score", linewidth=2, color="C1")

    # Running average of recent scores
    episode_scores = [convergence[i]["best_score"] for i in range(len(convergence))]
    # Also plot per-generation scores from search log if available
    try:
        with open(os.path.join(os.path.dirname(fig_dir), "g1_evolutionary", "search_log.json"), "r") as f:
            search_log = json.load(f)
        gen_scores = [s["episode_score"] for s in search_log]
        ax.plot(range(len(gen_scores)), gen_scores, alpha=0.3, color="C1", label="Per-gen Score")

        # Sliding average
        window = 10
        if len(gen_scores) >= window:
            smoothed = np.convolve(gen_scores, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(gen_scores)), smoothed,
                    label=f"Sliding Avg (w={window})", linewidth=2, color="C3")
    except Exception:
        pass

    ax.set_xlabel("Generation")
    ax.set_ylabel("Episode Score")
    ax.set_title("G1 Search Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "search_convergence.png"), dpi=300)
    plt.close(fig)
    print("  6/10 search_convergence.png")


def _plot_feature_importance(feat_imp: Dict[str, float], fig_dir: str) -> None:
    """7. EBM feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_items = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ["C3" if v > 0 else "C0" for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score")
    ax.set_title("EBM Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300)
    plt.close(fig)
    print("  7/10 feature_importance.png")


def _plot_cumulative_failures(
    g0_labels: np.ndarray, g1_labels: np.ndarray, fig_dir: str,
) -> None:
    """8. Cumulative failure count vs episode number."""
    fig, ax = plt.subplots(figsize=(8, 5))
    g0_cum = np.cumsum(g0_labels)
    g1_cum = np.cumsum(g1_labels)
    ax.plot(range(1, len(g0_cum) + 1), g0_cum, label="G0 Random", linewidth=2)
    ax.plot(range(1, len(g1_cum) + 1), g1_cum, label="G1 Evolutionary", linewidth=2)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Cumulative Failures")
    ax.set_title("Cumulative Failure Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "cumulative_failure_curve.png"), dpi=300)
    plt.close(fig)
    print("  8/10 cumulative_failure_curve.png")


def _plot_perturbation_space(
    g0_episodes: List[Dict], g1_episodes: List[Dict], fig_dir: str,
) -> None:
    """9. Perturbation space scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for episodes, name, marker in [
        (g0_episodes, "G0", "o"),
        (g1_episodes, "G1", "^"),
    ]:
        eps_s = [ep["eps_s"] for ep in episodes]
        p = [ep["p"] for ep in episodes]
        labels = [ep.get("failure_label", 0) for ep in episodes]

        fail_mask = np.array(labels) == 1
        succ_mask = ~fail_mask

        ax.scatter(
            np.array(eps_s)[succ_mask], np.array(p)[succ_mask],
            marker=marker, alpha=0.5, c="C0", label=f"{name} Success",
            edgecolors="black", linewidths=0.5, s=40,
        )
        ax.scatter(
            np.array(eps_s)[fail_mask], np.array(p)[fail_mask],
            marker=marker, alpha=0.8, c="C3", label=f"{name} Failure",
            edgecolors="black", linewidths=0.5, s=60,
        )

    ax.set_xlabel("Observation Noise (eps_s)")
    ax.set_ylabel("Action Replacement Prob (p)")
    ax.set_title("Perturbation Space")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "perturbation_space.png"), dpi=300)
    plt.close(fig)
    print("  9/10 perturbation_space.png")


def _plot_failure_coverage_heatmap(
    g0_episodes: List[Dict],
    g1_episodes: List[Dict],
    top_features: List[str],
    top_indices: List[int],
    quantile_boundaries: Dict[str, np.ndarray],
    fig_dir: str,
    n_bins: int = 3,
) -> None:
    """10. Failure coverage heatmap: 3x3 sub-grids for G0 and G1."""
    # Count failures per bin
    def count_failures_per_bin(episodes: List[Dict]) -> Dict:
        counts = {}
        for ep in episodes:
            if ep.get("failure_label", 0) == 1:
                bk = get_bin_index(ep, top_indices, top_features, quantile_boundaries, n_bins)
                counts[bk] = counts.get(bk, 0) + 1
        return counts

    g0_counts = count_failures_per_bin(g0_episodes)
    g1_counts = count_failures_per_bin(g1_episodes)

    max_count = max(
        max(g0_counts.values()) if g0_counts else 0,
        max(g1_counts.values()) if g1_counts else 0,
        1,
    )

    fig, axes = plt.subplots(2, n_bins, figsize=(12, 8))

    for row, (counts, title) in enumerate([
        (g0_counts, "G0 Random"),
        (g1_counts, "G1 Evolutionary"),
    ]):
        for k in range(n_bins):
            ax = axes[row, k]
            grid = np.zeros((n_bins, n_bins))
            for i in range(n_bins):
                for j in range(n_bins):
                    key = (i, j, k)
                    grid[i, j] = counts.get(key, 0)

            im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=max_count,
                          origin="lower", aspect="equal")

            # Add text annotations
            for i in range(n_bins):
                for j in range(n_bins):
                    val = int(grid[i, j])
                    color = "white" if val > max_count * 0.5 else "black"
                    ax.text(j, i, str(val), ha="center", va="center",
                           fontsize=12, fontweight="bold", color=color)

            ax.set_xticks(range(n_bins))
            ax.set_yticks(range(n_bins))
            if k == 0:
                ax.set_ylabel(f"{title}\n{top_features[1]}", fontsize=11)
            ax.set_xlabel(top_features[0], fontsize=10)
            ax.set_title(f"{top_features[2]} bin {k}", fontsize=11)

    fig.suptitle("Failure Coverage Heatmap", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Failure Count", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "failure_coverage_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  10/10 failure_coverage_heatmap.png")


if __name__ == "__main__":
    generate_all_figures()
