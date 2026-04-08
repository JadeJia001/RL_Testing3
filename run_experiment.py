"""Main entry point for the Model-Based RL Testing Framework."""

import argparse
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import omegaconf
import torch


def check_prerequisites(step: str) -> bool:
    """Check that required artifacts from prior steps exist."""
    checks = {
        "extract": [
            ("results/trained_model/all_episodes.pkl", "train"),
        ],
        "ebm": [
            ("results/statistics/feature_matrix.npy", "extract"),
            ("results/statistics/hazardous_labels.npy", "extract"),
        ],
        "g0": [
            ("results/trained_model/all_episodes.pkl", "train"),
            ("results/ebm_model/ebm_model.pkl", "ebm"),
            ("results/statistics/statistics.pkl", "extract"),
        ],
        "g1": [
            ("results/trained_model/all_episodes.pkl", "train"),
            ("results/ebm_model/ebm_model.pkl", "ebm"),
            ("results/g0_random/episodes.pkl", "g0"),
            ("results/statistics/statistics.pkl", "extract"),
        ],
        "evaluate": [
            ("results/g0_random/episodes.pkl", "g0"),
            ("results/g1_evolutionary/all_episodes.pkl", "g1"),
            ("results/ebm_model/feature_importance.json", "ebm"),
        ],
    }

    if step not in checks:
        return True

    for path, required_step in checks[step]:
        if not os.path.exists(path):
            print(f"ERROR: Missing {path}. Run --step {required_step} first.")
            return False
    return True


def load_dynamics_model(results_dir: str = "results/trained_model"):
    """Load trained dynamics model."""
    import mbrl.models
    import mbrl.planning
    import mbrl.util.common as common_util

    with open(os.path.join(results_dir, "model_cfg.pkl"), "rb") as f:
        cfg_dict = pickle.load(f)
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs_shape = env.observation_space.shape
    act_shape = (1,)

    dynamics_model = common_util.create_one_dim_tr_model(
        cfg, obs_shape, act_shape,
    )
    dynamics_model.load(results_dir)
    env.close()
    return dynamics_model, cfg


def load_agent(dynamics_model, results_dir: str = "results/trained_model"):
    """Reconstruct CEM agent from saved config."""
    import gymnasium as gym
    import mbrl.models
    import mbrl.planning
    from rl_testing.training.train_pets import (
        cartpole_termination_fn,
        cartpole_reward_fn,
    )

    with open(os.path.join(results_dir, "agent_cfg.pkl"), "rb") as f:
        agent_cfg_dict = pickle.load(f)
    agent_cfg = omegaconf.OmegaConf.create(agent_cfg_dict)

    env = gym.make("CartPole-v1")
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, cartpole_termination_fn, cartpole_reward_fn,
        generator=generator,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, agent_cfg, num_particles=5,
    )
    env.close()
    return agent, model_env


def step_train() -> None:
    """Step 1: Train PETS agent."""
    from rl_testing.training.train_pets import train_pets
    train_pets(results_dir="results/trained_model", seed=42)


def step_extract() -> None:
    """Step 2: Extract features and compute statistics."""
    from rl_testing.features.feature_extractor import (
        compute_and_save_statistics,
        extract_and_save_all,
    )

    print("=" * 60)
    print("Step 2: Feature Extraction")
    print("=" * 60)

    # Load training episodes
    with open("results/trained_model/all_episodes.pkl", "rb") as f:
        episodes = pickle.load(f)

    # Compute statistics
    print("Computing statistics from training data...")
    stats = compute_and_save_statistics(episodes, stats_dir="results/statistics")

    # Load dynamics model
    print("Loading dynamics model...")
    dynamics_model, _ = load_dynamics_model()

    # Extract features
    print("Extracting features for all training episodes...")
    extract_and_save_all(episodes, dynamics_model, stats, results_dir="results")


def step_ebm() -> None:
    """Step 3: Train EBM."""
    from rl_testing.ebm.train_ebm import train_ebm
    train_ebm(results_dir="results", seed=42)


def step_g0() -> None:
    """Step 4: Run G0 random search."""
    from rl_testing.search.random_search import run_g0_random_search

    # Load model and agent
    dynamics_model, _ = load_dynamics_model()
    agent, _ = load_agent(dynamics_model)

    # Load stats and EBM
    with open("results/statistics/statistics.pkl", "rb") as f:
        stats = pickle.load(f)
    with open("results/ebm_model/ebm_model.pkl", "rb") as f:
        ebm_model = pickle.load(f)

    run_g0_random_search(
        agent=agent,
        dynamics_model=dynamics_model,
        stats=stats,
        ebm_model=ebm_model,
        n_episodes=150,
        seed=42,
        results_dir="results/g0_random",
    )


def step_g1() -> None:
    """Step 5: Run G1 evolutionary search."""
    from rl_testing.search.evolutionary_search import run_g1_evolutionary_search

    # Load model and agent
    dynamics_model, _ = load_dynamics_model()
    agent, _ = load_agent(dynamics_model)

    # Load stats and EBM
    with open("results/statistics/statistics.pkl", "rb") as f:
        stats = pickle.load(f)
    with open("results/ebm_model/ebm_model.pkl", "rb") as f:
        ebm_model = pickle.load(f)

    run_g1_evolutionary_search(
        agent=agent,
        dynamics_model=dynamics_model,
        stats=stats,
        ebm_model=ebm_model,
        config_path="rl_testing/configs/search_config.yaml",
        g0_episodes_path="results/g0_random/episodes.pkl",
        results_dir="results/g1_evolutionary",
        seed=42,
    )


def step_evaluate() -> None:
    """Step 6: Evaluate and visualize."""
    from rl_testing.evaluation.evaluate import evaluate_all
    from rl_testing.evaluation.visualize import generate_all_figures

    evaluate_all(results_dir="results")
    generate_all_figures(results_dir="results")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-Based RL Testing Framework"
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["train", "extract", "ebm", "g0", "g1", "evaluate", "all"],
        help="Which step to run.",
    )
    args = parser.parse_args()

    steps = {
        "train": step_train,
        "extract": step_extract,
        "ebm": step_ebm,
        "g0": step_g0,
        "g1": step_g1,
        "evaluate": step_evaluate,
    }

    if args.step == "all":
        step_order = ["train", "extract", "ebm", "g0", "g1", "evaluate"]
    else:
        step_order = [args.step]

    for step_name in step_order:
        if not check_prerequisites(step_name):
            sys.exit(1)

        start_time = time.time()
        print(f"\n{'#' * 60}")
        print(f"# Starting: {step_name}")
        print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}\n")

        try:
            steps[step_name]()
        except Exception as e:
            print(f"\nERROR in step '{step_name}': {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        elapsed = time.time() - start_time
        print(f"\n  Step '{step_name}' completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
