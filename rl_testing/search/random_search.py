"""G0: Random search baseline - 150 episodes with random perturbation parameters."""

import json
import os
import pickle
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from rl_testing.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from rl_testing.oracle.fault_oracle import FaultOracle
from rl_testing.perturbation.perturbation import run_perturbed_episode


def run_g0_random_search(
    agent: Any,
    dynamics_model: Any,
    stats: Dict[str, Any],
    ebm_model: Any,
    n_episodes: int = 150,
    seed: int = 42,
    results_dir: str = "results/g0_random",
) -> List[Dict[str, Any]]:
    """Run G0 random search: 150 episodes with random perturbation params.

    Args:
        agent: PETS agent.
        dynamics_model: Trained dynamics model.
        stats: Statistics dict from feature extraction.
        ebm_model: Trained EBM model.
        n_episodes: Number of episodes to run.
        seed: Random seed.
        results_dir: Directory to save results.

    Returns:
        List of episode result dicts.
    """
    print("=" * 60)
    print("Step 4: G0 Random Search")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Sample perturbation parameters
    eps_s_values = rng.uniform(0.0, 0.5, size=n_episodes)
    p_values = rng.uniform(0.0, 0.3, size=n_episodes)

    # Setup
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
    all_episodes: List[Dict[str, Any]] = []

    for i in range(n_episodes):
        ep_seed = seed + i + 1000
        ep = run_perturbed_episode(agent, env, eps_s_values[i], p_values[i], ep_seed)

        # Extract features
        features = extractor.extract_features(ep)

        # EBM danger scores
        danger_scores = ebm_model.predict_proba(features)[:, 1]
        episode_score = float(np.max(danger_scores))

        # Fault oracle label
        failure_label = oracle.label_episode(ep)

        ep["features"] = features
        ep["danger_scores"] = danger_scores
        ep["episode_score"] = episode_score
        ep["failure_label"] = failure_label

        all_episodes.append(ep)

        if (i + 1) % 30 == 0:
            n_fail = sum(e["failure_label"] for e in all_episodes)
            print(f"  Episode {i + 1}/{n_episodes}, "
                  f"Failures: {n_fail}/{i + 1}, "
                  f"Score: {episode_score:.4f}")

    env.close()

    # Save results
    with open(os.path.join(results_dir, "episodes.pkl"), "wb") as f:
        pickle.dump(all_episodes, f)

    # Summary
    n_failures = sum(e["failure_label"] for e in all_episodes)
    failure_rate = n_failures / len(all_episodes)
    scores = [e["episode_score"] for e in all_episodes]
    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))

    summary = {
        "total_episodes": len(all_episodes),
        "failure_rate": failure_rate,
        "mean_episode_score": mean_score,
        "max_episode_score": max_score,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n  G0 Random Search Results:")
    print(f"  Total episodes: {len(all_episodes)}")
    print(f"  Failure rate: {failure_rate:.4f} ({n_failures}/{len(all_episodes)})")
    print(f"  Mean episode score: {mean_score:.4f}")
    print(f"  Max episode score: {max_score:.4f}")

    # Top 5 by score
    sorted_eps = sorted(all_episodes, key=lambda x: x["episode_score"], reverse=True)
    print(f"\n  Top 5 episodes by score:")
    for j, ep in enumerate(sorted_eps[:5]):
        print(f"    #{j+1}: score={ep['episode_score']:.4f}, "
              f"eps_s={ep['eps_s']:.3f}, p={ep['p']:.3f}, "
              f"failure={ep['failure_label']}")

    return all_episodes
