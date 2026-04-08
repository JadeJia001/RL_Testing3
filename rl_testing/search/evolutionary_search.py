"""G1: Diversity-aware evolutionary search with MAP-Elites archive."""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml

from rl_testing.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from rl_testing.oracle.fault_oracle import FaultOracle
from rl_testing.perturbation.perturbation import run_perturbed_episode


class Archive:
    """MAP-Elites style archive with multi-dimensional bins."""

    def __init__(
        self,
        top_feature_names: List[str],
        top_feature_indices: List[int],
        quantile_boundaries: Dict[str, np.ndarray],
        n_bins: int = 3,
        top_k: int = 5,
    ) -> None:
        self.top_feature_names = top_feature_names
        self.top_feature_indices = top_feature_indices
        self.quantile_boundaries = quantile_boundaries
        self.n_bins = n_bins
        self.top_k = top_k
        # bins[tuple of bin indices] -> list of episodes sorted by score desc
        self.bins: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}

    def get_bin_index(self, episode: Dict[str, Any]) -> Tuple[int, ...]:
        """Map episode to bin based on top-3 feature max values."""
        features = episode["features"]  # (T, 7)
        indices = []
        for fname, fidx in zip(self.top_feature_names, self.top_feature_indices):
            max_val = float(np.max(features[:, fidx]))
            boundaries = self.quantile_boundaries[fname]
            bin_idx = int(np.searchsorted(boundaries, max_val))
            bin_idx = min(bin_idx, self.n_bins - 1)
            indices.append(bin_idx)
        return tuple(indices)

    def try_add(self, episode: Dict[str, Any]) -> bool:
        """Try to add episode to archive. Returns True if added."""
        bin_key = self.get_bin_index(episode)
        score = episode["episode_score"]

        if bin_key not in self.bins:
            self.bins[bin_key] = []

        bin_list = self.bins[bin_key]

        if len(bin_list) < self.top_k:
            bin_list.append(episode)
            bin_list.sort(key=lambda x: x["episode_score"], reverse=True)
            return True
        elif score > bin_list[-1]["episode_score"]:
            bin_list[-1] = episode
            bin_list.sort(key=lambda x: x["episode_score"], reverse=True)
            return True

        return False

    def sample_parent(
        self, rng: np.random.Generator, tournament_size: int = 3,
        tournament_prob: float = 0.8,
    ) -> Dict[str, Any]:
        """Sample a parent: 80% tournament, 20% uniform random."""
        all_episodes = []
        for bin_list in self.bins.values():
            all_episodes.extend(bin_list)

        if not all_episodes:
            raise ValueError("Archive is empty")

        if rng.random() < tournament_prob:
            # Tournament selection
            candidates = rng.choice(
                len(all_episodes), size=min(tournament_size, len(all_episodes)),
                replace=False,
            )
            best_idx = max(candidates, key=lambda i: all_episodes[i]["episode_score"])
            return all_episodes[best_idx]
        else:
            # Uniform random
            return all_episodes[rng.integers(0, len(all_episodes))]

    def total_count(self) -> int:
        """Total number of episodes in archive."""
        return sum(len(v) for v in self.bins.values())

    def non_empty_bins(self) -> int:
        """Number of non-empty bins."""
        return len(self.bins)

    def failure_count(self) -> int:
        """Total failures in archive."""
        count = 0
        for bin_list in self.bins.values():
            count += sum(1 for ep in bin_list if ep.get("failure_label", 0) == 1)
        return count

    def print_bin_status(self) -> None:
        """Print bin occupancy."""
        total_bins = self.n_bins ** len(self.top_feature_names)
        print(f"  Archive: {self.total_count()} episodes in "
              f"{self.non_empty_bins()}/{total_bins} bins, "
              f"{self.failure_count()} failures")


def run_g1_evolutionary_search(
    agent: Any,
    dynamics_model: Any,
    stats: Dict[str, Any],
    ebm_model: Any,
    config_path: str = "rl_testing/configs/search_config.yaml",
    g0_episodes_path: str = "results/g0_random/episodes.pkl",
    results_dir: str = "results/g1_evolutionary",
    seed: int = 42,
) -> Tuple[Archive, List[Dict[str, Any]]]:
    """Run G1 diversity-aware evolutionary search.

    Args:
        agent: PETS agent.
        dynamics_model: Trained dynamics model.
        stats: Statistics dict.
        ebm_model: Trained EBM.
        config_path: Path to search config YAML.
        g0_episodes_path: Path to G0 episodes for initialization.
        results_dir: Directory to save results.
        seed: Random seed.

    Returns:
        (archive, all_episodes)
    """
    print("=" * 60)
    print("Step 5: G1 Evolutionary Search")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n_init = config["n_init_episodes"]
    n_generations = config["n_generations"]
    top_n_features = config["top_n_features"]
    n_bins = config["n_bins"]
    top_k = config["top_k_per_bin"]
    tournament_size = config["tournament_size"]
    tournament_prob = config["tournament_prob"]
    mutation_std = config["mutation_std"]

    rng = np.random.default_rng(seed)

    # Get top-3 features by importance
    feat_imp_path = os.path.join(os.path.dirname(results_dir), "ebm_model", "feature_importance.json")
    with open(feat_imp_path, "r") as f:
        feat_importance = json.load(f)

    sorted_features = sorted(feat_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [name for name, _ in sorted_features[:top_n_features]]
    top_indices = [FEATURE_NAMES.index(name) for name in top_features]
    print(f"  Top-{top_n_features} features: {top_features}")

    # Load quantile boundaries
    stats_dir = os.path.join(os.path.dirname(results_dir), "statistics")
    with open(os.path.join(stats_dir, "quantile_boundaries.pkl"), "rb") as f:
        quantile_boundaries = pickle.load(f)

    # Create archive
    archive = Archive(
        top_feature_names=top_features,
        top_feature_indices=top_indices,
        quantile_boundaries=quantile_boundaries,
        n_bins=n_bins,
        top_k=top_k,
    )

    # Load G0 first 50 episodes for initialization
    print(f"  Loading first {n_init} episodes from G0...")
    with open(g0_episodes_path, "rb") as f:
        g0_episodes = pickle.load(f)

    init_episodes = g0_episodes[:n_init]
    all_episodes = list(init_episodes)  # shallow copy

    # Add to archive
    for ep in init_episodes:
        archive.try_add(ep)

    print(f"  After initialization:")
    archive.print_bin_status()

    # Setup for new episodes
    oracle = FaultOracle(tau=stats["tau"], tau_step=stats["tau_step"])
    extractor = FeatureExtractor(
        dynamics_model=dynamics_model,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        reward_std=stats["reward_std"],
        reward_var=stats["reward_var"],
        train_states=stats["train_states"],
    )
    env = gym.make("CartPole-v1")

    search_log: List[Dict[str, Any]] = []
    convergence: List[Dict[str, Any]] = []
    best_score = max(ep["episode_score"] for ep in all_episodes)

    # Evolution loop
    print(f"\n  Running {n_generations} generations...")
    for gen in range(n_generations):
        # Select parent
        parent = archive.sample_parent(rng, tournament_size, tournament_prob)

        # Mutate
        eps_s_new = float(np.clip(parent["eps_s"] + mutation_std * rng.normal(), 0.0, 0.5))
        p_new = float(np.clip(parent["p"] + mutation_std * rng.normal(), 0.0, 0.3))

        # Run episode
        ep_seed = seed + n_init + gen + 2000
        ep = run_perturbed_episode(agent, env, eps_s_new, p_new, ep_seed)

        # Extract features and score
        features = extractor.extract_features(ep)
        danger_scores = ebm_model.predict_proba(features)[:, 1]
        episode_score = float(np.max(danger_scores))
        failure_label = oracle.label_episode(ep)

        ep["features"] = features
        ep["danger_scores"] = danger_scores
        ep["episode_score"] = episode_score
        ep["failure_label"] = failure_label

        # Try to add to archive
        added = archive.try_add(ep)
        bin_key = archive.get_bin_index(ep)

        all_episodes.append(ep)
        best_score = max(best_score, episode_score)

        # Log
        search_log.append({
            "generation": gen,
            "episode_score": episode_score,
            "bin": list(bin_key),
            "added": added,
            "failure": failure_label,
            "eps_s": eps_s_new,
            "p": p_new,
        })

        cum_failures = sum(1 for e in all_episodes if e.get("failure_label", 0) == 1)
        convergence.append({
            "generation": gen,
            "best_score": best_score,
            "cumulative_failures": cum_failures,
            "archive_size": archive.total_count(),
        })

        if (gen + 1) % 10 == 0:
            print(f"  Gen {gen + 1}/{n_generations}: "
                  f"score={episode_score:.4f}, "
                  f"archive={archive.total_count()}, "
                  f"failures={cum_failures}")

    env.close()

    # Save results
    with open(os.path.join(results_dir, "archive.pkl"), "wb") as f:
        pickle.dump(archive, f)

    with open(os.path.join(results_dir, "all_episodes.pkl"), "wb") as f:
        pickle.dump(all_episodes, f)

    with open(os.path.join(results_dir, "search_log.json"), "w") as f:
        json.dump(search_log, f, indent=2)

    with open(os.path.join(results_dir, "convergence.json"), "w") as f:
        json.dump(convergence, f, indent=2)

    # Final summary
    n_total = len(all_episodes)
    n_failures = sum(1 for e in all_episodes if e.get("failure_label", 0) == 1)
    total_bins = n_bins ** len(top_features)

    print(f"\n  G1 Evolutionary Search Results:")
    print(f"  Total episodes: {n_total}")
    print(f"  Archive: {archive.total_count()} in {archive.non_empty_bins()}/{total_bins} bins")
    print(f"  Failure rate: {n_failures/n_total:.4f} ({n_failures}/{n_total})")
    print(f"  Max episode score: {best_score:.4f}")

    return archive, all_episodes
