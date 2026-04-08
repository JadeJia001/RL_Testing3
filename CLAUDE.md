# CLAUDE.md

## Core Rules

### 1. Containerized Execution
- All code must run inside the container
- Run commands with: docker exec rl-testing bash -c "cd /workspace && ..."
- Install new dependencies inside the container and update requirements.txt

### 2. Experiment Integrity
- Results must be faithful to actual runs; never hardcode or beautify metrics
- Do not adjust thresholds, calculations, or re-run with different seeds to improve results
- Framework design (metric definitions, Oracle rules, search algorithms) is fixed
- If results are poor (low AUROC, low failure rate), report honestly

### 3. Code Standards
- All code and comments in English
- Use type hints
- Write docstrings for key functions
- Modular design, single responsibility per module
