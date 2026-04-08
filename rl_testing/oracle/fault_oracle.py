"""Fault Oracle: two-level episode labeling for CartPole-v1."""

from typing import Any, Dict, List, Tuple

import numpy as np


class FaultOracle:
    """Labels episodes as failure/success and steps as hazardous/safe."""

    def __init__(
        self,
        max_episode_steps: int = 500,
        horizon: int = 10,
        tau: float = 0.0,
        tau_step: float = 0.0,
    ) -> None:
        self.max_episode_steps = max_episode_steps
        self.horizon = horizon
        self.tau = tau
        self.tau_step = tau_step

    @staticmethod
    def compute_thresholds(
        episodes: List[Dict[str, Any]],
        max_episode_steps: int = 500,
        horizon: int = 10,
    ) -> Tuple[float, float]:
        """Compute tau (episode-level) and tau_step from training episodes.

        Args:
            episodes: List of episode dicts with 'episode_return', 'rewards', etc.
            max_episode_steps: Max steps per episode.
            horizon: Lookahead horizon for step-level labels.

        Returns:
            (tau, tau_step) thresholds.
        """
        # Episode-level: tau = bottom 10% episode return
        returns = [ep["episode_return"] for ep in episodes]
        tau = float(np.percentile(returns, 10))

        # Step-level: tau_step = bottom 20% of H-horizon tail returns
        horizon_returns: List[float] = []
        for ep in episodes:
            rewards = ep["rewards"]
            T = len(rewards)
            for t in range(T):
                end = min(t + horizon, T)
                tail_return = float(np.sum(rewards[t:end]))
                horizon_returns.append(tail_return)

        tau_step = float(np.percentile(horizon_returns, 20))

        return tau, tau_step

    def label_episode(self, episode: Dict[str, Any]) -> int:
        """Label episode as failure (1) or success (0).

        Primary: terminated early (steps < max_episode_steps).
        Fallback: cumulative reward < tau.
        """
        ep_len = episode["episode_length"]
        dones = episode["dones"]

        # Check if terminated early (not truncated)
        if ep_len < self.max_episode_steps and dones[-1]:
            return 1

        # Fallback: low reward
        if episode["episode_return"] < self.tau:
            return 1

        return 0

    def label_steps(self, episode: Dict[str, Any]) -> np.ndarray:
        """Label each step as hazardous (1) or safe (0).

        Step t is hazardous if cumulative reward from t to min(t+H, T) < tau_step.
        """
        rewards = episode["rewards"]
        T = len(rewards)
        labels = np.zeros(T, dtype=np.int32)

        for t in range(T):
            end = min(t + self.horizon, T)
            tail_return = np.sum(rewards[t:end])
            if tail_return < self.tau_step:
                labels[t] = 1

        return labels
