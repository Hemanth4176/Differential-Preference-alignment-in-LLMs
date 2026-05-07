# RLVR Project README

## Overview
This repository contains a research implementation of **RLVR (Reinforcement Learning with Verifiable Rewards)** for solving math problems from the GSM8K dataset. The system supports three training modes:
- **GRADE** – differentiable proxy reward using Gumbel‑Softmax.
- **GRPO** – exact non‑differentiable reward via verification.
- **Hybrid** – combines both GRADE and GRPO.

The core components are:
- `analysis_script.py` – generates visualizations and statistical reports from training results.
- `training_grade.py` – implements the training loop, data handling, proxy verifier, and policy model.
- `main.py` – orchestrates training for each mode and runs the analysis after completion.

## Scripts
### `analysis_script.py`
- Loads result JSON files from `results/<mode>/` directories.
- Provides a suite of 12 Matplotlib figures (reward, loss, KL divergence, proxy reward, trust factor, temperature schedule, gradient analysis, test accuracy, running accuracy, combined dashboard, GRPO‑specific metrics, hybrid decomposition).
- Generates a statistical analysis report (`statistical_analysis.txt`).
- Utilises helper functions for smoothing, answer extraction, and plotting aesthetics.

### `training_grade.py`
- Defines configuration dataclass `RLVRConfig` with model, LoRA, training, and RLVR hyper‑parameters.
- Manages device allocation via `DeviceManager` singleton (policy, reference, proxy devices).
- Handles GSM8K dataset loading, splits for proxy training, policy training, validation, and testing.
- Implements differentiable generator using Gumbel‑Softmax top‑k and gradient checkpointing.
- Provides `ProxyVerifier` model that approximates the exact binary reward for GRADE training.
- Includes functions to generate proxy training data, train the proxy verifier, and compute rewards.
- Contains the main training loop (`training_main`) that performs policy updates, proxy updates, evaluation, early‑stopping checks, and checkpoint saving.

### `main.py`
- Sets output directory and evaluation interval.
- Configures CUDA memory stability settings.
- Defines `cleanup_between_modes` to reset CUDA state and device manager after each training run.
- Runs training for `grade_only` and `grpo_only` modes sequentially, handling exceptions.
- After successful runs, invokes `analysis_main` from `analysis_script.py` to produce figures and reports.

## Requirements
- Python 3.10+ (tested on Windows).
- PyTorch with CUDA support.
- `transformers`, `datasets`, `peft`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `wandb` (disabled by default).

Install dependencies via:
```bash
pip install torch transformers datasets peft tqdm numpy scipy matplotlib tqdm
```

## Usage
1. **Prepare data** – place the GSM8K Arrow dataset at the path configured in `RLVRConfig.local_dataset_path`.
2. **Run training** – execute:
```bash
python main.py
```
   This will train GRADE and GRPO modes sequentially, saving results under `./results`.
3. **Analyze results** – after training completes (or if you only want analysis), run:
```bash
python analysis_script.py --results_dir ./results
```
   Figures are saved to `results/figures` and a textual report to `results/figures/statistical_analysis.txt`.

## Notes
- The code includes extensive safety checks for CUDA health, early‑stopping based on KL collapse and extraction rate, and automatic proxy verifier re‑training.
- Adjust hyper‑parameters in `RLVRConfig` as needed for different hardware or dataset sizes.
- `wandb` logging is disabled; enable by setting `WANDB_MODE` environment variable.

