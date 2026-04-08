"""Evaluation: compute metrics for G0 and G1."""

import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from rl_testing.features.feature_extractor import FEATURE_NAMES


def compute_ece(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error.

    Args:
        scores: Predicted scores (probabilities).
        labels: Binary labels.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(scores)

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_score = scores[mask].mean()
            actual_rate = labels[mask].mean()
            ece += (n_bin / N) * abs(avg_score - actual_rate)

    return float(ece)


def get_bin_index(
    episode: Dict[str, Any],
    top_feature_indices: List[int],
    top_feature_names: List[str],
    quantile_boundaries: Dict[str, np.ndarray],
    n_bins: int = 3,
) -> Tuple[int, ...]:
    """Map an episode to a bin index."""
    features = episode["features"]
    indices = []
    for fname, fidx in zip(top_feature_names, top_feature_indices):
        max_val = float(np.max(features[:, fidx]))
        boundaries = quantile_boundaries[fname]
        bin_idx = int(np.searchsorted(boundaries, max_val))
        bin_idx = min(bin_idx, n_bins - 1)
        indices.append(bin_idx)
    return tuple(indices)


def compute_failure_coverage(
    episodes: List[Dict[str, Any]],
    top_feature_indices: List[int],
    top_feature_names: List[str],
    quantile_boundaries: Dict[str, np.ndarray],
    n_bins: int = 3,
) -> Tuple[int, int, float]:
    """Compute failure coverage metrics.

    Returns:
        (failure_bins_hit, total_bins_touched, coverage_ratio)
    """
    failure_bins = set()
    all_bins = set()

    for ep in episodes:
        bin_key = get_bin_index(
            ep, top_feature_indices, top_feature_names,
            quantile_boundaries, n_bins,
        )
        all_bins.add(bin_key)
        if ep.get("failure_label", 0) == 1:
            failure_bins.add(bin_key)

    total_possible = n_bins ** len(top_feature_names)
    failure_bins_hit = len(failure_bins)
    total_bins_touched = len(all_bins)
    coverage_ratio = failure_bins_hit / max(total_bins_touched, 1)

    return failure_bins_hit, total_possible, coverage_ratio


def evaluate_all(
    results_dir: str = "results",
) -> Dict[str, Dict[str, Any]]:
    """Evaluate G0 and G1, print comparison table.

    Args:
        results_dir: Base results directory.

    Returns:
        Dict with G0 and G1 metrics.
    """
    print("=" * 60)
    print("Step 6: Evaluation")
    print("=" * 60)

    # Load episodes
    with open(os.path.join(results_dir, "g0_random", "episodes.pkl"), "rb") as f:
        g0_episodes = pickle.load(f)
    with open(os.path.join(results_dir, "g1_evolutionary", "all_episodes.pkl"), "rb") as f:
        g1_episodes = pickle.load(f)

    # Load feature importance for top-3
    with open(os.path.join(results_dir, "ebm_model", "feature_importance.json"), "r") as f:
        feat_imp = json.load(f)
    sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [name for name, _ in sorted_features[:3]]
    top_indices = [FEATURE_NAMES.index(name) for name in top_features]

    # Load quantile boundaries
    with open(os.path.join(results_dir, "statistics", "quantile_boundaries.pkl"), "rb") as f:
        quantile_boundaries = pickle.load(f)

    results = {}
    for name, episodes in [("G0", g0_episodes), ("G1", g1_episodes)]:
        scores = np.array([ep["episode_score"] for ep in episodes])
        labels = np.array([ep.get("failure_label", 0) for ep in episodes])

        n_total = len(episodes)
        n_failures = int(labels.sum())
        failure_rate = n_failures / n_total

        # AUROC, AUPRC
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
        else:
            auroc = float("nan")
            auprc = float("nan")

        # ECE
        ece = compute_ece(scores, labels)

        # Time to first failure
        ttff = None
        for i, ep in enumerate(episodes):
            if ep.get("failure_label", 0) == 1:
                ttff = i + 1
                break

        # Max / Mean score
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))

        # Failure coverage
        fb_hit, total_bins, cov_ratio = compute_failure_coverage(
            episodes, top_indices, top_features, quantile_boundaries,
        )

        results[name] = {
            "total_episodes": n_total,
            "failure_rate": failure_rate,
            "n_failures": n_failures,
            "failure_coverage_bins": fb_hit,
            "failure_coverage_total": total_bins,
            "failure_coverage_ratio": cov_ratio,
            "auroc": auroc,
            "auprc": auprc,
            "ece": ece,
            "time_to_first_failure": ttff,
            "max_episode_score": max_score,
            "mean_episode_score": mean_score,
            "scores": scores,
            "labels": labels,
        }

    # Print comparison table
    table = format_comparison_table(results)
    print(table)

    # Save report
    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("Model-Based RL Testing Framework - Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(table)

    print(f"\n  Report saved to: {report_path}")

    # Save metrics as JSON (without numpy arrays)
    metrics_json = {}
    for name in ["G0", "G1"]:
        m = {k: v for k, v in results[name].items() if k not in ("scores", "labels")}
        # Convert NaN to None for JSON
        for k, v in m.items():
            if isinstance(v, float) and np.isnan(v):
                m[k] = None
        metrics_json[name] = m

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    return results


def format_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format a comparison table for G0 vs G1."""
    g0 = results["G0"]
    g1 = results["G1"]

    def fmt(val: Any) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            if np.isnan(val):
                return "N/A"
            return f"{val:.4f}"
        return str(val)

    lines = [
        "| Metric                          | G0 Random   | G1 Evolutionary |",
        "|--------------------------------|-------------|-----------------|",
        f"| Total episodes                 | {g0['total_episodes']:>11} | {g1['total_episodes']:>15} |",
        f"| Failure rate                   | {fmt(g0['failure_rate']):>11} | {fmt(g1['failure_rate']):>15} |",
        f"| Failure coverage (bins hit)    | {g0['failure_coverage_bins']:>5}/27     | {g1['failure_coverage_bins']:>9}/27     |",
        f"| Failure coverage (ratio)       | {fmt(g0['failure_coverage_ratio']):>11} | {fmt(g1['failure_coverage_ratio']):>15} |",
        f"| AUROC                          | {fmt(g0['auroc']):>11} | {fmt(g1['auroc']):>15} |",
        f"| AUPRC                          | {fmt(g0['auprc']):>11} | {fmt(g1['auprc']):>15} |",
        f"| ECE                            | {fmt(g0['ece']):>11} | {fmt(g1['ece']):>15} |",
        f"| Time-to-first-failure          | {fmt(g0['time_to_first_failure']):>11} | {fmt(g1['time_to_first_failure']):>15} |",
        f"| Max episode_score              | {fmt(g0['max_episode_score']):>11} | {fmt(g1['max_episode_score']):>15} |",
        f"| Mean episode_score             | {fmt(g0['mean_episode_score']):>11} | {fmt(g1['mean_episode_score']):>15} |",
    ]
    return "\n".join(lines) + "\n"
