"""Perturbation strategy for testing PETS agent."""

from typing import Any, Dict

import gymnasium as gym
import numpy as np


def run_perturbed_episode(
    agent: Any,
    env: gym.Env,
    eps_s: float,
    p: float,
    seed: int,
) -> Dict[str, Any]:
    """Run a single perturbed episode.

    Args:
        agent: PETS agent with .act() and .reset() methods.
        env: CartPole-v1 environment.
        eps_s: Observation noise scale.
        p: Action replacement probability.
        seed: Random seed for this episode.

    Returns:
        Episode dict with states, actions, rewards, etc.
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    agent.reset()

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    episode_return = 0.0
    step = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        states.append(obs.copy())

        # Add observation noise
        noise = rng.normal(0, 1, size=obs.shape).astype(np.float32)
        obs_perturbed = obs + eps_s * noise

        # Agent decides action based on perturbed observation
        action_continuous = agent.act(obs_perturbed)
        action_int = 1 if action_continuous.item() >= 0.5 else 0

        # With probability p, replace with random action
        if rng.random() < p:
            action_int = rng.integers(0, 2)

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action_int)

        actions.append(action_int)
        rewards.append(reward)
        next_states.append(next_obs.copy())
        dones.append(terminated or truncated)

        obs = next_obs
        episode_return += reward
        step += 1

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_states": np.array(next_states),
        "dones": np.array(dones),
        "episode_return": episode_return,
        "episode_length": step,
        "eps_s": eps_s,
        "p": p,
        "seed": seed,
    }
