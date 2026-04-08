"""Train PETS agent on CartPole-v1 using mbrl-lib."""

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch

import mbrl.models
import mbrl.planning
import mbrl.util.common as common_util


def cartpole_termination_fn(
    act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    """CartPole termination function for model-based planning."""
    assert len(next_obs.shape) == 2
    x = next_obs[:, 0]
    theta = next_obs[:, 2]
    done = (x.abs() > 2.4) | (theta.abs() > 0.2094)
    done = done[:, None]
    return done


def cartpole_reward_fn(
    act: torch.Tensor, next_obs: torch.Tensor
) -> torch.Tensor:
    """CartPole reward function: +1 for every step alive."""
    return torch.ones(next_obs.shape[0], 1, device=next_obs.device)


def train_pets(
    results_dir: str = "results/trained_model",
    seed: int = 42,
) -> None:
    """Train PETS on CartPole-v1 and save model + transition data."""
    print("=" * 60)
    print("Step 1: Training PETS on CartPole-v1")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cpu")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Create environment
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape
    act_shape = (1,)  # CartPole discrete -> we treat as continuous for CEM

    # Config for mbrl-lib
    cfg = omegaconf.OmegaConf.create({
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": str(device),
            "num_layers": 3,
            "ensemble_size": 5,
            "hid_size": 128,
            "deterministic": False,
            "propagation_method": "fixed_model",
            "learn_logvar_bounds": False,
        },
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        "overrides": {
            "num_elites": 5,
            "trial_length": 500,
            "num_steps": 500 * 200,
        },
    })

    # Create dynamics model
    dynamics_model = common_util.create_one_dim_tr_model(
        cfg, obs_shape, act_shape
    )

    # Create replay buffer
    replay_buffer = common_util.create_replay_buffer(
        cfg, obs_shape, act_shape, rng=rng
    )

    # Model trainer
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=1e-3,
        weight_decay=5e-5,
    )

    # Agent config
    agent_cfg = omegaconf.OmegaConf.create({
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 10,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": [0.0],
        "action_ub": [1.0],
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 3,
            "elite_ratio": 0.1,
            "population_size": 50,
            "alpha": 0.1,
            "return_mean_elites": True,
            "clipped_normal": False,
            "lower_bound": "???",
            "upper_bound": "???",
            "device": str(device),
        },
    })

    # Create model environment
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, cartpole_termination_fn, cartpole_reward_fn,
        generator=generator,
    )

    # Create agent
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, agent_cfg, num_particles=5
    )

    # Collect initial random data
    print("Collecting initial random data (1000 steps)...")
    common_util.rollout_agent_trajectories(
        env,
        steps_or_trials_to_collect=1000,
        agent=mbrl.planning.RandomAgent(env),
        agent_kwargs={},
        trial_length=500,
        replay_buffer=replay_buffer,
    )
    print(f"  Replay buffer size: {len(replay_buffer)}")

    # Initial model training
    print("Initial model training...")
    dynamics_model.update_normalizer(replay_buffer.get_all())
    train_dataset, val_dataset = common_util.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=256,
        val_ratio=0.2,
        ensemble_size=5,
        shuffle_each_epoch=True,
        bootstrap_permutes=False,
    )
    model_trainer.train(
        train_dataset, dataset_val=val_dataset, num_epochs=20,
        patience=20, improvement_threshold=0.01, silent=True,
    )

    # Training loop: collect episodes with PETS agent
    all_episodes: List[Dict[str, Any]] = []
    episode_returns: List[float] = []
    num_trials = 50

    print(f"Training PETS for {num_trials} episodes...")
    for trial in range(num_trials):
        obs, _ = env.reset(seed=seed + trial + 1)
        agent.reset()

        states = [obs.copy()]
        actions = []
        rewards = []
        next_states = []
        dones = []
        terminated_flag = False
        truncated_flag = False
        episode_return = 0.0
        step = 0

        while not (terminated_flag or truncated_flag):
            # Agent acts: CEM outputs continuous action, threshold at 0.5
            action_continuous = agent.act(obs)
            action_int = 1 if action_continuous.item() >= 0.5 else 0

            next_obs, reward, terminated_flag, truncated_flag, info = env.step(action_int)

            # Store transition
            actions.append(action_int)
            rewards.append(reward)
            next_states.append(next_obs.copy())
            dones.append(terminated_flag or truncated_flag)

            # Add to replay buffer (need to format for mbrl)
            action_array = np.array([action_continuous.item()], dtype=np.float32)
            replay_buffer.add(obs, action_array, next_obs, reward, terminated_flag, truncated_flag)

            obs = next_obs
            states.append(obs.copy())
            episode_return += reward
            step += 1

        episode_data = {
            "states": np.array(states[:-1]),  # exclude last extra state
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
            "episode_return": episode_return,
            "episode_length": step,
        }
        all_episodes.append(episode_data)
        episode_returns.append(episode_return)

        # Retrain model periodically
        if (trial + 1) % 5 == 0:
            dynamics_model.update_normalizer(replay_buffer.get_all())
            train_dataset, val_dataset = common_util.get_basic_buffer_iterators(
                replay_buffer,
                batch_size=256,
                val_ratio=0.2,
                ensemble_size=5,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,
            )
            model_trainer.train(
                train_dataset, dataset_val=val_dataset, num_epochs=20,
                patience=20, improvement_threshold=0.01, silent=True,
            )

        if (trial + 1) % 10 == 0:
            recent_avg = np.mean(episode_returns[-10:])
            print(f"  Episode {trial + 1}/{num_trials}, "
                  f"Return: {episode_return:.0f}, "
                  f"Avg(last 10): {recent_avg:.1f}")

    env.close()

    # Save results
    print("Saving trained model and data...")

    # Save dynamics model
    dynamics_model.save(results_dir)

    # Save agent config for later reconstruction
    with open(os.path.join(results_dir, "agent_cfg.pkl"), "wb") as f:
        pickle.dump(omegaconf.OmegaConf.to_container(agent_cfg), f)

    # Save model config
    with open(os.path.join(results_dir, "model_cfg.pkl"), "wb") as f:
        pickle.dump(omegaconf.OmegaConf.to_container(cfg), f)

    # Save all episodes
    with open(os.path.join(results_dir, "all_episodes.pkl"), "wb") as f:
        pickle.dump(all_episodes, f)

    # Save episode returns
    np.save(os.path.join(results_dir, "episode_returns.npy"), np.array(episode_returns))

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_returns, alpha=0.3, label="Episode Return")
    window = 20
    if len(episode_returns) >= window:
        smoothed = np.convolve(episode_returns, np.ones(window)/window, mode="valid")
        plt.plot(range(window - 1, len(episode_returns)), smoothed,
                 label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PETS Training on CartPole-v1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curve.png"), dpi=300)
    plt.close()

    # Print summary
    print(f"\nTraining complete:")
    print(f"  Total episodes: {len(all_episodes)}")
    print(f"  Final avg return (last 20): {np.mean(episode_returns[-20:]):.1f}")
    print(f"  Max return: {max(episode_returns):.0f}")
    print(f"  Episodes reaching 500: {sum(1 for r in episode_returns if r >= 500)}")
    print(f"  Results saved to: {results_dir}/")


if __name__ == "__main__":
    train_pets()
