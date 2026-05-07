# RLVR Project

## Overview

This repository contains a research implementation of **RLVR (Reinforcement Learning with Verifiable Rewards)** for solving math reasoning problems from the GSM8K dataset.

The project supports three training modes:

- **GRADE** — Differentiable proxy reward using Gumbel-Softmax.
- **GRPO** — Exact non-differentiable reward via verification.
- **Hybrid** — Combines both GRADE and GRPO.

Core components:

- `analysis_script.py` — Generates visualizations and statistical reports from training results.
- `training_grade.py` — Implements the training loop, proxy verifier, policy model, and RLVR pipeline.
- `main.py` — Orchestrates training runs and post-training analysis.

---

# GRADE-STE for RLVR
## Replacing Policy Gradients with Backpropagation for Math Reasoning

This project adapts the **GRADE-STE** framework from sentiment control to **Reinforcement Learning with Verifiable Rewards (RLVR)** on GSM8K.

## Key Changes from the Original GRADE-STE Implementation

1. **Dataset**
   - IMDB sentiment → GSM8K math reasoning

2. **Reward**
   - Neural sentiment classifier → Verifiable answer correctness

3. **Differentiability Bridge**
   - Train a proxy verifier that approximates the binary correctness signal
   - Enables backpropagation through Gumbel-Softmax

4. **Hybrid Training**
   - Combines:
     - GRADE (differentiable proxy)
     - GRPO (exact verifier)

5. **Longer Sequence Generation**
   - 64 tokens → 512+ tokens for chain-of-thought reasoning

6. **Answer Extraction**
   - Parses:
     - `\boxed{...}`
     - `#### <number>`

---

# Architecture Overview

```text
┌──────────────┐
│  Prompt (x)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Policy LLM (πθ)     │
│  (LoRA fine-tuned)   │
└──────┬───────────────┘
       │
       │ soft tokens ỹ (differentiable)
       ▼
   ┌────┴────┐
   │         │
   ▼         ▼
┌────────┐ ┌─────────────────────┐
│ Proxy  │ │ Exact Verifier      │
│Verifier│ │ (non-differentiable)│
│ (diff) │ │ extract_answer(ỹ)   │
│  R̂(ỹ) │ │ == ground_truth?    │
└───┬────┘ └─────────┬───────────┘
    │                │
    │ ∇θ R̂          │ binary reward
    │ (backprop)     │ (GRPO / logging)
    ▼                ▼
GRADE loss       GRPO loss
(optional hybrid training)
```

---

# Repository Structure

## `analysis_script.py`

Generates visualizations and statistical reports from training outputs.

### Features

- Loads result JSON files from:
  - `results/<mode>/`
- Produces:
  - Reward curves
  - Loss curves
  - KL divergence plots
  - Proxy reward analysis
  - Trust factor plots
  - Temperature schedules
  - Gradient analysis
  - Test accuracy curves
  - Running accuracy
  - Combined dashboards
  - GRPO-specific metrics
  - Hybrid decomposition plots
- Generates:
  - `statistical_analysis.txt`

### Output

Saved under:

```text
results/figures/
```

---

## `training_grade.py`

Implements the full RLVR training pipeline.

### Main Components

#### `RLVRConfig`
Configuration dataclass containing:

- Model settings
- LoRA configuration
- Training hyperparameters
- RLVR parameters

#### `DeviceManager`
Singleton for device allocation:

- Policy model device
- Reference model device
- Proxy verifier device

#### Dataset Pipeline

Handles:

- GSM8K loading
- Proxy training split
- Policy training split
- Validation split
- Test split

#### Differentiable Generation

Implements:

- Gumbel-Softmax top-k sampling
- Gradient checkpointing

#### `ProxyVerifier`

A neural verifier trained to approximate the exact binary reward.

#### Training Utilities

Includes:

- Proxy training data generation
- Reward computation
- Proxy verifier training
- Evaluation utilities
- Checkpointing

#### Main Training Loop

`training_main`

Supports:

- GRADE training
- GRPO training
- Hybrid optimization
- Policy updates
- Proxy updates
- Early stopping
- Evaluation
- Automatic checkpoint saving

---

## `main.py`

Main orchestration script.

### Responsibilities

- Sets output directories
- Configures CUDA memory stability
- Sequentially runs:
  - `grade_only`
  - `grpo_only`
- Handles cleanup between runs
- Launches analysis after training

### CUDA Stability Features

Uses:

```python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

to reduce fragmentation-induced CUDA OOM issues.

---

# Requirements

- Python 3.10+
- CUDA-enabled PyTorch

## Python Packages

- `torch`
- `transformers`
- `datasets`
- `peft`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`
- `wandb` *(optional)*

---

# Installation

```bash
pip install torch transformers datasets peft tqdm numpy scipy matplotlib
```

---

# Usage

## 1. Prepare the Dataset

Place the GSM8K Arrow dataset at the location specified in:

```python
RLVRConfig.local_dataset_path
```

---

## 2. Run Training

```bash
python main.py
```

This sequentially trains:

- GRADE
- GRPO

Results are stored in:

```text
./results
```

---

## 3. Run Analysis Separately

```bash
python analysis_script.py --results_dir ./results
```

Generated outputs:

- Figures → `results/figures`
- Statistical report → `results/figures/statistical_analysis.txt`

---

# Training Modes

| Mode | Description |
|---|---|
| `grade_only` | Differentiable reward optimization using proxy verifier |
| `grpo_only` | Exact RLVR optimization using binary verification |
| `hybrid` | Combines proxy gradients with exact verifier rewards |

---

# Safety and Stability Features

The codebase includes:

- CUDA health monitoring
- Automatic memory cleanup
- Early stopping based on:
  - KL collapse
  - Extraction rate degradation
- Automatic proxy verifier retraining
- Gradient checkpointing
- Trust-factor scheduling

---

# Logging

`wandb` logging is disabled by default.

To enable:

```bash
export WANDB_MODE=online
```

or on Windows:

```powershell
set WANDB_MODE=online
```

---

# Output Structure

```text
results/
├── grade_only/
├── grpo_only/
├── hybrid/
└── figures/
    ├── *.png
    └── statistical_analysis.txt
```

---

# Research Focus

This project investigates whether:

- policy-gradient RL can be partially replaced by
- differentiable proxy optimization

for:

- long-horizon reasoning
- chain-of-thought generation
- verifiable mathematical rewards

using:

- Gumbel-Softmax relaxation
- proxy reward modeling
- hybrid RL optimization.

---
