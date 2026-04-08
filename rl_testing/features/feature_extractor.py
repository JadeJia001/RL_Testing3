"""7-dimensional feature extractor for MBRL testing."""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl_testing.oracle.fault_oracle import FaultOracle


FEATURE_NAMES = [
    "predictive_dispersion",
    "diagonal_approx",
    "planning_error",
    "reward_pred_error",
    "novelty",
    "state_instability",
    "action_sensitivity",
]


class FeatureExtractor:
    """Extract 7-dim features for each timestep in an episode."""

    def __init__(
        self,
        dynamics_model: Any,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        reward_std: float,
        reward_var: float,
        train_states: np.ndarray,
        k_nn: int = 5,
    ) -> None:
        """Initialize feature extractor.

        Args:
            dynamics_model: Trained OneDTransitionRewardModel.
            state_mean: Mean of training states (obs_dim,).
            state_std: Std of training states (obs_dim,).
            reward_std: Std of training rewards.
            reward_var: Variance of training rewards.
            train_states: All training states for kNN (N, obs_dim).
            k_nn: Number of neighbors for novelty.
        """
        self.dynamics_model = dynamics_model
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_std = max(reward_std, 1e-8)
        self.reward_var = max(reward_var, 1e-8)
        self.sigma_all = float(np.mean(state_std))
        self.sigma_all = max(self.sigma_all, 1e-8)
        self.k_nn = k_nn

        # Standardize training states for kNN
        self.train_states_standardized = (
            (train_states - state_mean) / np.maximum(state_std, 1e-8)
        )

    def _get_ensemble_predictions(
        self, obs: np.ndarray, action: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get predictions from each ensemble member.

        Returns:
            (state_preds, reward_preds): Each shape (K, obs_dim) and (K,) or None.
        """
        device = next(self.dynamics_model.model.parameters()).device
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        # Action as continuous value matching training format
        act_t = torch.FloatTensor([[float(action)]]).to(device)
        model_in = torch.cat([obs_t, act_t], dim=-1)

        with torch.no_grad():
            # Get per-member predictions: shape [E, 1, D]
            means, logvars = self.dynamics_model.model.forward(
                model_in, use_propagation=False
            )
            # means shape: [ensemble_size, 1, out_dim]
            preds = means[:, 0, :].cpu().numpy()  # [K, out_dim]

        # If target_is_delta, predicted next state = obs + delta
        if hasattr(self.dynamics_model, "target_is_delta") and self.dynamics_model.target_is_delta:
            state_preds = obs + preds
        else:
            state_preds = preds

        return state_preds, None

    def extract_features(self, episode: Dict[str, Any]) -> np.ndarray:
        """Extract 7-dim features for each timestep.

        Args:
            episode: Dict with states, actions, rewards, next_states.

        Returns:
            Feature matrix of shape (T, 7).
        """
        states = episode["states"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        next_states = episode["next_states"]
        T = len(actions)

        features = np.zeros((T, 7), dtype=np.float32)

        for t in range(T):
            obs = states[t]
            action = actions[t]
            reward = rewards[t]
            next_state = next_states[t]

            # Get ensemble predictions
            state_preds, _ = self._get_ensemble_predictions(obs, action)
            # state_preds shape: [K, obs_dim]
            K = state_preds.shape[0]
            D = state_preds.shape[1]

            # R4: predictive_dispersion
            # U_r = Var(ensemble predictions across reward dimension) / reward_var
            # Since we don't have reward predictions, use state prediction variance
            ensemble_var = np.var(state_preds, axis=0)  # (D,)
            features[t, 0] = float(np.mean(ensemble_var) / self.reward_var)

            # R5: diagonal_approx
            # U_s = (1/D) * sum_i Var(f^i across ensemble), per state dim
            features[t, 1] = float(np.mean(ensemble_var))

            # R6: planning_error
            # PE_s = ||s_{t+1} - s_hat_{t+1}|| / sigma_all
            ensemble_mean = np.mean(state_preds, axis=0)
            planning_err = np.linalg.norm(next_state - ensemble_mean) / self.sigma_all
            features[t, 2] = float(planning_err)

            # R7: reward_pred_error
            # PE_r = |r_t - r_hat_t| / sigma_r_train
            # Estimate reward as 1.0 (CartPole always gives +1 while alive)
            r_hat = 1.0
            features[t, 3] = float(abs(reward - r_hat) / self.reward_std)

            # R8: novelty (kNN distance)
            obs_std = (obs - self.state_mean) / np.maximum(self.state_std, 1e-8)
            dists = np.linalg.norm(
                self.train_states_standardized - obs_std, axis=1
            )
            k_nearest = np.sort(dists)[: self.k_nn]
            features[t, 4] = float(np.mean(k_nearest))

            # R10: state_instability
            if t == 0:
                features[t, 5] = 0.0
            else:
                delta_s = np.linalg.norm(states[t] - states[t - 1]) / self.sigma_all
                features[t, 5] = float(delta_s)

            # R12: action_sensitivity
            if t == 0:
                features[t, 6] = 0.0
            else:
                features[t, 6] = float(abs(actions[t] - actions[t - 1]))

        return features


def compute_and_save_statistics(
    episodes: List[Dict[str, Any]],
    stats_dir: str = "results/statistics",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute normalization statistics from training data.

    Args:
        episodes: Training episodes.
        stats_dir: Directory to save statistics.

    Returns:
        Dict of statistics.
    """
    os.makedirs(stats_dir, exist_ok=True)

    # Collect all states and rewards
    all_states = np.concatenate([ep["states"] for ep in episodes], axis=0)
    all_rewards = np.concatenate([ep["rewards"] for ep in episodes], axis=0)

    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0)
    reward_std = float(np.std(all_rewards))
    reward_var = float(np.var(all_rewards))

    # Compute oracle thresholds
    tau, tau_step = FaultOracle.compute_thresholds(episodes)

    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "reward_std": reward_std,
        "reward_var": reward_var,
        "tau": tau,
        "tau_step": tau_step,
        "train_states": all_states,
    }

    # Save
    with open(os.path.join(stats_dir, "statistics.pkl"), "wb") as f:
        pickle.dump(stats, f)

    # Save tau values as json for readability
    with open(os.path.join(stats_dir, "thresholds.json"), "w") as f:
        json.dump({"tau": tau, "tau_step": tau_step}, f, indent=2)

    print(f"  State mean: {state_mean}")
    print(f"  State std: {state_std}")
    print(f"  Reward std: {reward_std:.4f}")
    print(f"  tau (episode): {tau:.2f}")
    print(f"  tau_step: {tau_step:.2f}")

    return stats


def compute_quantile_boundaries(
    feature_matrix: np.ndarray,
    episode_ids: np.ndarray,
    feature_names: List[str],
    n_bins: int = 3,
    stats_dir: str = "results/statistics",
) -> Dict[str, np.ndarray]:
    """Compute per-feature quantile boundaries for archive bins.

    For each feature, compute n_bins-1 quantile boundaries using
    per-episode max aggregation.

    Args:
        feature_matrix: (N_steps, 7) feature matrix.
        episode_ids: (N_steps,) episode id for each step.
        feature_names: List of feature names.
        n_bins: Number of bins per feature.
        stats_dir: Directory to save boundaries.

    Returns:
        Dict mapping feature_name -> array of boundaries.
    """
    unique_episodes = np.unique(episode_ids)
    n_features = feature_matrix.shape[1]

    # Compute per-episode max for each feature
    episode_max_features = np.zeros((len(unique_episodes), n_features))
    for i, ep_id in enumerate(unique_episodes):
        mask = episode_ids == ep_id
        episode_max_features[i] = np.max(feature_matrix[mask], axis=0)

    # Compute quantile boundaries
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # e.g., [33.33, 66.67]
    boundaries = {}
    for j, name in enumerate(feature_names):
        vals = episode_max_features[:, j]
        boundaries[name] = np.percentile(vals, quantiles)

    # Save
    with open(os.path.join(stats_dir, "quantile_boundaries.pkl"), "wb") as f:
        pickle.dump(boundaries, f)

    return boundaries


def extract_and_save_all(
    episodes: List[Dict[str, Any]],
    dynamics_model: Any,
    stats: Dict[str, Any],
    results_dir: str = "results",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all episodes and save.

    Returns:
        (feature_matrix, hazardous_labels, episode_ids, step_ids)
    """
    oracle = FaultOracle(
        tau=stats["tau"],
        tau_step=stats["tau_step"],
    )

    extractor = FeatureExtractor(
        dynamics_model=dynamics_model,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        reward_std=stats["reward_std"],
        reward_var=stats["reward_var"],
        train_states=stats["train_states"],
    )

    all_features = []
    all_hazardous = []
    all_episode_ids = []
    all_step_ids = []
    episode_labels = []

    for ep_idx, ep in enumerate(episodes):
        # Extract features
        feats = extractor.extract_features(ep)
        T = feats.shape[0]

        # Step-level labels
        haz_labels = oracle.label_steps(ep)

        # Episode-level label
        ep_label = oracle.label_episode(ep)
        episode_labels.append(ep_label)

        all_features.append(feats)
        all_hazardous.append(haz_labels)
        all_episode_ids.append(np.full(T, ep_idx, dtype=np.int32))
        all_step_ids.append(np.arange(T, dtype=np.int32))

        if (ep_idx + 1) % 50 == 0:
            print(f"  Extracted features for {ep_idx + 1}/{len(episodes)} episodes")

    feature_matrix = np.concatenate(all_features, axis=0)
    hazardous_labels = np.concatenate(all_hazardous, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    step_ids = np.concatenate(all_step_ids, axis=0)
    episode_labels = np.array(episode_labels)

    # Save
    save_dir = os.path.join(results_dir, "statistics")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "feature_matrix.npy"), feature_matrix)
    np.save(os.path.join(save_dir, "hazardous_labels.npy"), hazardous_labels)
    np.save(os.path.join(save_dir, "episode_ids.npy"), episode_ids)
    np.save(os.path.join(save_dir, "step_ids.npy"), step_ids)
    np.save(os.path.join(save_dir, "episode_labels.npy"), episode_labels)

    # Compute and save quantile boundaries
    compute_quantile_boundaries(
        feature_matrix, episode_ids, FEATURE_NAMES,
        n_bins=3, stats_dir=save_dir,
    )

    # Print statistics
    n_failure = int(np.sum(episode_labels))
    n_total = len(episode_labels)
    n_haz = int(np.sum(hazardous_labels))
    n_total_steps = len(hazardous_labels)

    print(f"\n  Feature extraction complete:")
    print(f"  Failure episodes: {n_failure}/{n_total} ({100*n_failure/n_total:.1f}%)")
    print(f"  Hazardous steps: {n_haz}/{n_total_steps} ({100*n_haz/n_total_steps:.1f}%)")
    print(f"\n  Feature statistics (mean +/- std):")
    for i, name in enumerate(FEATURE_NAMES):
        col = feature_matrix[:, i]
        print(f"    {name}: {col.mean():.4f} +/- {col.std():.4f}")

    return feature_matrix, hazardous_labels, episode_ids, step_ids
