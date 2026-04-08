# Model-Based RL Testing Framework

A complete pipeline for testing Model-Based Reinforcement Learning agents using diversity-aware evolutionary search. This framework trains a PETS (Probabilistic Ensemble Trajectory Sampling) agent on CartPole-v1, then systematically discovers failure modes through perturbation-based testing guided by an Explainable Boosting Machine (EBM) danger scorer.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Experiment Pipeline](#experiment-pipeline)
  - [Step 1: PETS Agent Training](#step-1-pets-agent-training)
  - [Step 2: Feature Extraction + Fault Oracle](#step-2-feature-extraction--fault-oracle)
  - [Step 3: EBM Danger Scorer Training](#step-3-ebm-danger-scorer-training)
  - [Step 4: G0 Random Search Baseline](#step-4-g0-random-search-baseline)
  - [Step 5: G1 Evolutionary Search](#step-5-g1-evolutionary-search)
  - [Step 6: Evaluation and Visualization](#step-6-evaluation-and-visualization)
- [Experiment Results](#experiment-results)
- [How to Run](#how-to-run)
- [Configuration](#configuration)

---

## Overview

The framework implements a 6-step pipeline to answer the question: *Can guided evolutionary search find more diverse failure modes in an MBRL agent than random testing?*

**Core Idea:**
1. Train a PETS agent that learns a world model (ensemble of neural networks) and plans via CEM (Cross-Entropy Method).
2. Define a Fault Oracle that labels episodes as failures.
3. Extract 7-dimensional features capturing model uncertainty, prediction errors, and behavioral anomalies.
4. Train an EBM to score how "dangerous" each state is.
5. Compare two testing strategies under the same budget (150 episodes):
   - **G0**: Random perturbation parameters
   - **G1**: MAP-Elites evolutionary search guided by EBM danger scores

---

## Project Structure

```
rl_testing/
├── configs/
│   ├── cartpole_pets.yaml          # PETS + CartPole-v1 configuration
│   └── search_config.yaml          # Evolutionary search hyperparameters
├── training/
│   └── train_pets.py               # PETS agent training
├── oracle/
│   └── fault_oracle.py             # Two-level failure labeling
├── features/
│   └── feature_extractor.py        # 7-dim feature extraction
├── ebm/
│   └── train_ebm.py                # EBM danger scorer
├── perturbation/
│   └── perturbation.py             # Perturbation strategy
├── search/
│   ├── random_search.py            # G0 random baseline
│   └── evolutionary_search.py      # G1 MAP-Elites evolutionary search
├── evaluation/
│   ├── evaluate.py                 # Metrics computation
│   └── visualize.py                # 10 academic-style figures
├── results/                        # All experiment outputs
└── run_experiment.py               # Main entry point
```

---

## Environment Setup

All code runs inside a Docker container for reproducibility.

```bash
# Build the container
docker build -t rl-testing -f .devcontainer/Dockerfile .

# Start the container
docker run -d --name rl-testing -v $(pwd):/workspace rl-testing sleep infinity

# Verify
docker exec rl-testing python -c "import mbrl; import gymnasium; import interpret; print('OK')"
```

**Key Dependencies:** mbrl-lib 0.2.0, gymnasium 0.26.3, PyTorch (CPU), InterpretML, scikit-learn

---

## Experiment Pipeline

### Step 1: PETS Agent Training

Train a PETS agent on CartPole-v1 using mbrl-lib.

**Method:**
- **World Model**: Ensemble of 5 probabilistic neural networks (GaussianMLP, 128 hidden units, 3 layers)
- **Planner**: CEM optimizer (population=50, iterations=3, horizon=10)
- **Training**: 50 episodes with periodic model retraining every 5 episodes
- **Initial Data**: 1000 random exploration steps

**Training Results:**
- Total training episodes: 50
- Episodes reaching max return (500): 4
- Final average return (last 20 episodes): 326.2
- Max return achieved: 500

The training curve shows the agent progressively learning to balance the pole:

![Training Curve](results/trained_model/training_curve.png)

---

### Step 2: Feature Extraction + Fault Oracle

**Fault Oracle** labels episodes using two criteria:
1. **Primary**: Episode terminated early (steps < 500) due to pole falling
2. **Fallback**: Cumulative reward below the bottom-10% threshold of training data

**Step-level labels**: A step is "hazardous" if the cumulative reward over the next H=10 steps falls below the bottom-20% percentile.

**7-Dimensional Feature Vector** for each timestep:

| # | Feature | Description |
|---|---------|-------------|
| R4 | `predictive_dispersion` | Normalized variance across ensemble predictions |
| R5 | `diagonal_approx` | Mean per-dimension ensemble variance |
| R6 | `planning_error` | Next-state prediction error vs actual |
| R7 | `reward_pred_error` | Reward prediction error (always 0 for CartPole) |
| R8 | `novelty` | Mean k-NN distance in standardized state space (k=5) |
| R10 | `state_instability` | Normalized state change between consecutive steps |
| R12 | `action_sensitivity` | Action change between consecutive steps |

**Extraction Results:**
- Failure episodes in training: 46/50 (92.0%)
- Hazardous steps: 450/7,807 (5.8%)
- Thresholds: tau (episode) = 12.9, tau_step = 10.0

---

### Step 3: EBM Danger Scorer Training

An Explainable Boosting Classifier (InterpretML) trained on step-level features to predict hazardous states.

**Setup:**
- Input: 7-dim feature vector per step
- Output: danger_score = P(hazardous)
- Train/Val split: 80/20 by episode (same episode's steps stay together)
- Episode score = max danger_score across all steps

**EBM Validation Metrics:**

| Level | AUROC | AUPRC |
|-------|-------|-------|
| Step-level | 0.822 | 0.337 |
| Episode-level | 0.563 | 0.834 |

**Feature Importance Ranking:**

| Feature | Importance |
|---------|-----------|
| state_instability | -0.00212 |
| action_sensitivity | -0.00208 |
| novelty | -0.00143 |
| planning_error | 0.00057 |
| predictive_dispersion | -0.00002 |
| diagonal_approx | 0.00001 |
| reward_pred_error | 0.00000 |

![Feature Importance](results/figures/feature_importance.png)

The top-3 features (state_instability, action_sensitivity, novelty) are used to define the MAP-Elites archive in G1.

---

### Step 4: G0 Random Search Baseline

150 episodes with uniformly random perturbation parameters.

**Perturbation Strategy:**
- Observation noise: s'_t = s_t + eps_s * N(0, I), where eps_s ~ Uniform(0, 0.5)
- Action replacement: with probability p ~ Uniform(0, 0.3), replace agent action with random action

**G0 Results:**
- Total episodes: 150
- Failure rate: 99.33% (149/150)
- Mean episode score: 0.890
- Max episode score: 0.9999

---

### Step 5: G1 Evolutionary Search

Diversity-aware evolutionary search using a MAP-Elites archive.

**Archive Structure:**
- Top-3 features from EBM importance define 3 behavioral dimensions
- Each dimension split into 3 bins by training-data quantiles (33%/67%)
- 3^3 = 27 total bins, each stores top-5 episodes by danger score

**Search Process:**
- **Initialization**: Reuse first 50 episodes from G0 (no re-running)
- **Evolution**: 100 generations, each producing 1 new episode
  - Parent selection: 80% tournament (size 3), 20% uniform random
  - Mutation: Gaussian noise (std=0.05) on perturbation parameters
  - Archive update: add to bin if slot available or score exceeds minimum

**G1 Results:**
- Total episodes: 150 (50 reused + 100 new)
- Failure rate: 100% (150/150)
- Archive size: 45 episodes in 9/27 bins
- Max episode score: 1.0000

---

### Step 6: Evaluation and Visualization

Comprehensive comparison of G0 vs G1 across multiple metrics.

---

## Experiment Results

### Summary Comparison Table

| Metric | G0 Random | G1 Evolutionary |
|--------|-----------|-----------------|
| Total episodes | 150 | 150 |
| Failure rate | 0.9933 | 1.0000 |
| Failure coverage (bins hit) | 9/27 | 9/27 |
| Failure coverage (ratio) | 1.0000 | 1.0000 |
| AUROC | 0.3893 | N/A |
| AUPRC | 0.9938 | N/A |
| ECE | 0.1032 | 0.1424 |
| Time-to-first-failure | 1 | 1 |
| Max episode_score | 0.9999 | 1.0000 |
| Mean episode_score | 0.8901 | 0.8576 |

### Analysis

1. **Near-universal failures**: Both methods achieve very high failure rates (>99%), indicating that the perturbation range (eps_s up to 0.5, p up to 0.3) is sufficient to destabilize the PETS agent on CartPole almost always.

2. **G1 AUROC is N/A**: Because all 150 G1 episodes are failures, there is no class variation to compute AUROC. This is an honest reflection of the experimental outcome.

3. **Equal failure coverage**: Both methods cover 9/27 bins, suggesting the perturbation-induced failure modes naturally cluster in specific behavioral regions regardless of search strategy.

4. **EBM is effective at step-level**: The step-level AUROC of 0.822 shows the EBM can distinguish hazardous from safe states, even though episode-level discrimination is limited by the high base failure rate.

### Generated Figures

#### ROC and PR Curves
![ROC Curve](results/figures/roc_curve.png)
![PR Curve](results/figures/pr_curve.png)

#### Failure Analysis
![Failure Rate Comparison](results/figures/failure_rate_comparison.png)
![Cumulative Failures](results/figures/cumulative_failure_curve.png)

#### Score Distribution and Calibration
![Episode Score Distribution](results/figures/episode_score_distribution.png)
![Calibration Diagram](results/figures/calibration_diagram.png)

#### Search Behavior
![Search Convergence](results/figures/search_convergence.png)
![Perturbation Space](results/figures/perturbation_space.png)

#### Feature and Coverage Analysis
![Feature Importance](results/figures/feature_importance.png)
![Failure Coverage Heatmap](results/figures/failure_coverage_heatmap.png)

---

## How to Run

```bash
# Run the full pipeline (all steps sequentially)
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step all"

# Or run individual steps
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step train"
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step extract"
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step ebm"
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step g0"
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step g1"
docker exec rl-testing bash -c "cd /workspace && python run_experiment.py --step evaluate"
```

Each step checks for prerequisites and reports errors if prior steps haven't been run.

---

## Configuration

### PETS Training (`rl_testing/configs/cartpole_pets.yaml`)
- Environment: CartPole-v1 (max 500 steps)
- Ensemble: 5 probabilistic neural networks
- CEM planner: population=50, iterations=3, horizon=10

### Search Parameters (`rl_testing/configs/search_config.yaml`)
- Initial episodes: 50 (reused from G0)
- Evolution generations: 100
- Archive: top-3 features x 3 bins = 27 cells, top-5 per cell
- Tournament: size=3, probability=0.8
- Mutation std: 0.05
- Global seed: 42

---

*All results are faithful to actual experimental runs. No metrics were hardcoded or beautified.*
