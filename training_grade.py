"""
GRADE-STE for RLVR: Replacing Policy Gradients with Backpropagation for Math Reasoning
========================================================================================

Adapts the GRADE-STE framework from sentiment control (differentiable reward model)
to Reinforcement Learning with Verifiable Rewards (RLVR) on GSM8K.

Key changes from the original GRADE-STE implementation:
1. Dataset: IMDB sentiment → GSM8K math reasoning
2. Reward: Neural sentiment classifier → Verifiable answer correctness
3. Differentiability bridge: Train a proxy verifier (neural) that approximates
   the binary correctness signal, enabling backpropagation through Gumbel-Softmax
4. Hybrid training: Combine GRADE (differentiable proxy) + GRPO (exact verifier)
5. Longer sequences: 64 tokens → 512+ tokens for chain-of-thought reasoning
6. Answer extraction: Parse \\boxed{} or "#### <number>" from generated text

Architecture overview:
  ┌──────────────┐
  │  Prompt (x)  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────┐
  │  Policy LLM (πθ)     │  ← Gumbel-Softmax soft generation
  │  (LoRA fine-tuned)   │
  └──────┬───────────────┘
         │ soft tokens ỹ (differentiable)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 ┌────────┐ ┌─────────────────────┐
 │ Proxy  │ │ Exact Verifier      │
 │Verifier│ │ (non-differentiable)│
 │ (diff) │ │ extract_answer(ỹ)   │
 │ R̂(ỹ)  │ │   == ground_truth?  │
 └───┬────┘ └─────────┬───────────┘
     │                │
     │ ∇θ R̂          │ binary reward (for GRPO / logging)
     │ (backprop)     │
     ▼                ▼
  GRADE loss       GRPO loss (optional hybrid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import re
import json
from pathlib import Path
from collections import defaultdict
import os
# GPU 2 has uncorrectable ECC errors — exclude it.
# Physical GPUs 0,1,3 are mapped to logical cuda:0, cuda:1, cuda:2 by CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"  # skip faulty GPU 2

os.environ["WANDB_MODE"] = "disabled"


class DeviceManager:
    """
    Manages device assignments for multi-GPU setups.
    Spreads models across available GPUs to save memory:
      - policy  → cuda:0
      - ref     → cuda:1  (falls back to cuda:0 if only 1 GPU)
      - proxy   → cuda:2  (falls back to cuda:0)
    All tensor transfers between models go through explicit .to(device) calls.

    Lazy-initialised: device resolution happens on first attribute access,
    NOT at import time (so CUDA_VISIBLE_DEVICES is already processed).
    """
    _instance = None          # singleton
    _initialised = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _lazy_init(self):
        """Resolve devices on first use, not at import."""
        if self._initialised:
            return
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n == 0:
            self.policy = torch.device("cpu")
            self.ref    = torch.device("cpu")
            self.proxy  = torch.device("cpu")
        elif n == 1:
            self.policy = torch.device("cuda:0")
            self.ref    = torch.device("cuda:0")
            self.proxy  = torch.device("cuda:0")
        elif n == 2:
            self.policy = torch.device("cuda:0")
            self.ref    = torch.device("cuda:1")
            self.proxy  = torch.device("cuda:0")
        elif n == 3:
            # 3 healthy GPUs: spread policy / ref / proxy across all three
            self.policy = torch.device("cuda:0")
            self.ref    = torch.device("cuda:1")
            self.proxy  = torch.device("cuda:2")
        else:  # 4+ GPUs
            self.policy = torch.device("cuda:0")
            self.ref    = torch.device("cuda:1")
            self.proxy  = torch.device("cuda:2")
        self._initialised = True
        # Verify each device is healthy with a small allocation test
        for name, dev in [("policy", self.policy), ("ref", self.ref), ("proxy", self.proxy)]:
            if dev.type == "cuda":
                try:
                    _ = torch.zeros(1, device=dev)
                except RuntimeError as e:
                    print(f"  ⚠ WARNING: {name} device {dev} failed health check: {e}")
                    print(f"  Falling back to cuda:0 for {name}")
                    setattr(self, name, torch.device("cuda:0"))
        print(f"DeviceManager: policy={self.policy}, ref={self.ref}, proxy={self.proxy} ({n} GPUs)")

    def __getattr__(self, name):
        # Intercept attribute access to trigger lazy init
        if name in ("policy", "ref", "proxy"):
            self._lazy_init()
            return self.__dict__[name]
        raise AttributeError(name)


# Lazy singleton — devices resolved on first access, not import time
DEVICES = DeviceManager()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RLVRConfig:
    """Configuration for GRADE-STE adapted to RLVR on GSM8K."""

    # Model
    base_model: str = "/home/cccp/25m2118/RND2/Qwen2.5-3B-Instruct"      # Local path to the base model

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 3e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 2000
    eval_every: int = 200
    warmup_steps: int = 100

    # Generation — longer than sentiment because math needs chain-of-thought
    max_new_tokens: int = 256
    min_new_tokens: int = 32

    # Gumbel-Softmax
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_steps: int = 3000
    gumbel_topk: int = 256

    # KL regularization — increased from 0.05 to prevent distribution collapse
    kl_coef: float = 0.3
    # KL lower bound: penalize if KL drops below this to prevent collapse
    kl_lower_bound: float = 1.0
    kl_lower_coef: float = 5.0               # Penalty strength for KL below lower bound

    # GRPO (for hybrid mode)
    # Reduced from 32 → 8 to prevent CUDA OOM on A100 80GB
    grpo_group_size: int = 8
    grpo_clip: float = 0.2
    grpo_micro_batch: int = 4                 # Generate this many completions at a time

    # Hybrid training weights
    grade_weight: float = 0.5         # Weight for GRADE (proxy) loss
    grpo_weight: float = 0.5          # Weight for GRPO (exact) loss
    training_mode: str = "grade_only" # "grade_only", "grpo_only", "hybrid"

    # Proxy verifier
    proxy_verifier_lr: float = 1e-5
    proxy_verifier_retrain_every: int = 500   # Retrain more often to track policy shifts
    proxy_train_samples: int = 3000           # Samples for proxy verifier training
    proxy_min_positive_rate: float = 0.10     # Min fraction of correct samples in proxy data (raised from 0.05)
    proxy_data_batch_size: int = 8            # Larger batch for faster proxy data generation
    proxy_replay_buffer_size: int = 8000      # Replay buffer size for stable proxy training

    # Degeneracy detection
    degeneracy_penalty_coef: float = 2.0      # Penalize repetitive/degenerate outputs
    min_unique_token_ratio: float = 0.1       # Below this ratio → degenerate output

    # Early stopping thresholds
    kl_collapse_threshold: float = 0.5        # Stop if KL drops below this
    extraction_rate_threshold: float = 0.3    # Stop if extraction rate drops below this
    collapse_patience: int = 3                # Consecutive eval failures before stopping
    max_recovery_attempts: int = 2            # How many times to try recovery before final stop

    # Infrastructure
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./results"

    # Data
    dataset_name: str = None
    dataset_config: str = None
    max_prompt_length: int = 256
    val_size: int = 200
    local_dataset_path: str = "/home/cccp/25m2118/RND2/gsm8k_dataset"
# ============================================================================
# ANSWER EXTRACTION AND VERIFICATION
# ============================================================================

def extract_answer_gsm8k(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from a GSM8K-style response.

    Handles multiple formats:
      - "#### 42"        (GSM8K ground truth format)
      - "\\boxed{42}"    (LaTeX format used by many models)
      - "The answer is 42"
      - Last number in the response as fallback
    """
    # Try #### format (GSM8K standard)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Try \boxed{} format
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try "the answer is X" format
    match = re.search(r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Fallback: last number in the text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()

    return None


def extract_ground_truth_gsm8k(answer_text: str) -> str:
    """Extract ground truth from GSM8K answer field (after ####)."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return answer_text.strip()


def verify_answer(predicted: Optional[str], ground_truth: str) -> float:
    """
    Binary verification: does the predicted answer match ground truth?

    Returns 1.0 for correct, 0.0 for incorrect.
    This is the NON-DIFFERENTIABLE reward used in standard RLVR.
    """
    if predicted is None:
        return 0.0
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        return 1.0 if abs(pred_val - gt_val) < 1e-5 else 0.0
    except ValueError:
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

class GSM8KDataset(Dataset):
    """Wraps GSM8K for prompt-based training."""

    def __init__(self, data, tokenizer, max_prompt_length: int, include_answer: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.include_answer = include_answer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer_text = item["answer"]
        ground_truth = extract_ground_truth_gsm8k(answer_text)

        # Format as chat / instruction prompt
        prompt = (
            f"Solve the following math problem step by step. "
            f"Show your work and put your final answer after ####.\n\n"
            f"Question: {question}\n\n"
            f"Solution:"
        )

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "ground_truth": ground_truth,
            "question": question,
        }

        if self.include_answer:
            result["full_answer"] = answer_text

        return result


def collate_fn(batch):
    """Custom collate that handles string fields."""
    result = {}
    result["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    result["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    result["ground_truth"] = [b["ground_truth"] for b in batch]
    result["question"] = [b["question"] for b in batch]
    if "full_answer" in batch[0]:
        result["full_answer"] = [b["full_answer"] for b in batch]
    return result


class RLVRDataSplits:
    """
    Manages data splits for RLVR on GSM8K.

    GSM8K has:
      - train: 7,473 problems
      - test:  1,319 problems

    We split train into:
      - proxy_train: First proxy_train_samples for training the proxy verifier
      - policy_train: Remaining for policy training
      - val: Last val_size from policy_train for validation
    """

    def __init__(self, config: RLVRConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        print("\n" + "="*60)
        print("LOADING AND SPLITTING DATA (RLVR)")
        print("="*60)

        # ============================================================
        # Load dataset (Arrow format)
        # ============================================================
        full_dataset = load_from_disk(self.config.local_dataset_path)

        train_data = full_dataset["train"].shuffle(seed=config.seed)
        test_data = full_dataset["test"]

        n_train = len(train_data)

        # Safety checks
        if n_train == 0:
            raise ValueError("Training dataset is empty!")
        if len(test_data) == 0:
            raise ValueError("Test dataset is empty!")

        # ============================================================
        # RLVR SPLIT LOGIC
        # ============================================================

        proxy_size = min(self.config.proxy_train_samples, int(0.4 * n_train))
        val_size = min(self.config.val_size, n_train // 5)

        policy_start = proxy_size
        policy_end = n_train - val_size
        val_start = policy_end

        assert policy_end > policy_start, "Policy training split is empty!"
        assert val_start < n_train, "Validation split invalid!"

        # ============================================================
        # Create splits
        # ============================================================

        self.proxy_train_data = train_data.select(range(0, proxy_size))
        self.policy_train_data = train_data.select(range(policy_start, policy_end))
        self.val_data = train_data.select(range(val_start, n_train))
        self.test_data = test_data

        # ============================================================
        # Logging
        # ============================================================

        print(f"  Proxy Verifier Training: {len(self.proxy_train_data)} samples (0 → {proxy_size})")
        print(f"  Policy Training:         {len(self.policy_train_data)} samples ({policy_start} → {policy_end})")
        print(f"  Validation:              {len(self.val_data)} samples ({val_start} → {n_train})")
        print(f"  Test:                    {len(self.test_data)} samples (GSM8K test split)")
        print("="*60 + "\n")

    def get_proxy_train_dataloader(self, batch_size: int) -> DataLoader:
        dataset = GSM8KDataset(
            self.proxy_train_data, self.tokenizer,
            self.config.max_prompt_length, include_answer=True,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_policy_train_dataloader(self, batch_size: int) -> DataLoader:
        dataset = GSM8KDataset(
            self.policy_train_data, self.tokenizer,
            self.config.max_prompt_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        dataset = GSM8KDataset(
            self.val_data, self.tokenizer,
            self.config.max_prompt_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    def get_test_dataloader(self, batch_size: int) -> DataLoader:
        dataset = GSM8KDataset(
            self.test_data, self.tokenizer,
            self.config.max_prompt_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# ============================================================================
# GUMBEL-SOFTMAX UTILITIES
# All helper tensors (noise, zeros) are created via *_like or with explicit
# device= so they always land on the same device as the input logits.
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    # rand_like / zeros_like inherit device & dtype from `logits`
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


def gumbel_softmax_topk(
    logits: torch.Tensor, tau: float = 1.0, k: int = 256, hard: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient Gumbel-Softmax with top-k filtering.
    Device-safe: every tensor is created on logits.device.
    """
    dev = logits.device  # pin device once
    topk_logits, topk_indices = logits.topk(k, dim=-1)  # inherits dev
    gumbels = -torch.log(-torch.log(
        torch.rand(topk_logits.shape, device=dev, dtype=topk_logits.dtype) + 1e-10
    ) + 1e-10)
    y_soft_topk = F.softmax((topk_logits + gumbels) / tau, dim=-1)
    y_soft = torch.zeros(logits.shape, device=dev, dtype=logits.dtype).scatter_(
        -1, topk_indices, y_soft_topk
    )

    if hard:
        local_argmax = y_soft_topk.argmax(dim=-1, keepdim=True)
        global_argmax = topk_indices.gather(-1, local_argmax)
        y_hard = torch.zeros(logits.shape, device=dev, dtype=logits.dtype).scatter_(
            -1, global_argmax, 1.0
        )
        return (y_hard - y_soft).detach() + y_soft, topk_indices

    return y_soft, topk_indices


# ============================================================================
# DIFFERENTIABLE GENERATOR (adapted for longer sequences)
# ============================================================================

class DifferentiableGenerator(nn.Module):
    """
    Generates soft token sequences via Gumbel-Softmax for backprop through
    the reward model. Adapted from the original with:
    - Gradient checkpointing for longer sequences
    - Memory-efficient top-k generation as default
    """

    def __init__(self, model: nn.Module, tokenizer, config: RLVRConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding = model.get_input_embeddings()

    def get_tau(self, step: int) -> float:
        if step >= self.config.tau_anneal_steps:
            return self.config.tau_end
        ratio = step / self.config.tau_anneal_steps
        return self.config.tau_start - ratio * (self.config.tau_start - self.config.tau_end)

    def _policy_forward_step(self, policy_embeds, policy_mask):
        """Single-step forward pass, compatible with gradient checkpointing."""
        outputs = self.model(
            inputs_embeds=policy_embeds,
            attention_mask=policy_mask,
            use_cache=False,
        )
        return outputs.logits[:, -1, :]

    def generate_soft_topk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tau: float,
        topk: int = 256,
        use_ste: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Memory-efficient soft generation with top-k Gumbel-Softmax.

        Returns:
            topk_indices: [batch, seq, k]
            topk_weights: [batch, seq, k] — differentiable
            hard_tokens:  [batch, seq]     — argmax tokens for decoding
            logits_seq:   [batch, seq, vocab]
        """
        batch_size = input_ids.shape[0]
        device = DEVICES.policy

        policy_embeds = self.embedding(input_ids)
        policy_mask = attention_mask.clone()

        topk_indices_list = []
        topk_weights_list = []
        hard_tokens_list = []
        logits_list = []

        for _ in range(self.config.max_new_tokens):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                policy_logits = torch.utils.checkpoint.checkpoint(
                    self._policy_forward_step,
                    policy_embeds,
                    policy_mask,
                    use_reentrant=False,
                ).float()

            logits_list.append(policy_logits)

            soft_token, topk_idx = gumbel_softmax_topk(
                policy_logits, tau=tau, k=topk, hard=use_ste
            )

            topk_weights = soft_token.gather(-1, topk_idx)
            topk_indices_list.append(topk_idx.detach())
            topk_weights_list.append(topk_weights)

            hard_token = soft_token.argmax(dim=-1)
            hard_tokens_list.append(hard_token)

            next_embed = (
                soft_token.to(self.embedding.weight.dtype) @ self.embedding.weight
            ).unsqueeze(1)
            policy_embeds = torch.cat([policy_embeds, next_embed], dim=1)
            policy_mask = torch.cat([
                policy_mask, torch.ones(batch_size, 1, device=device)
            ], dim=1)

            del policy_logits, soft_token

        topk_indices = torch.stack(topk_indices_list, dim=1)
        topk_weights = torch.stack(topk_weights_list, dim=1)
        hard_tokens = torch.stack(hard_tokens_list, dim=1)
        logits_seq = torch.stack(logits_list, dim=1)

        return topk_indices, topk_weights, hard_tokens, logits_seq

    def generate_hard(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      max_new_tokens: Optional[int] = None) -> torch.Tensor:
        """Standard discrete generation for evaluation and GRPO sampling."""
        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )


# ============================================================================
# PROXY VERIFIER (differentiable approximation of exact verification)
# ============================================================================

class ProxyVerifier(nn.Module):
    """
    A neural network that approximates the binary verification function.

    Takes a generated solution (as soft token embeddings) and predicts
    P(correct). This makes the reward differentiable, enabling GRADE-STE.

    The proxy is trained on (solution, correctness_label) pairs generated
    by the base/current policy and scored by the exact verifier.

    Architecture:
      - Shares the same embedding space as the policy LLM
      - Transformer backbone (frozen) + classification head (trained)
      - Outputs scalar reward ∈ [0, 1]

    Key difference from original SameVocabRewardModel:
      - Trained on correctness labels, not sentiment
      - Input is [prompt + solution], not just [text]
      - Must be periodically retrained as policy distribution shifts
    """

    def __init__(self, base_model_name: str, device: str):
        super().__init__()
        resolved_device = torch.device(device)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=None,  # no auto-split
        )
        self.embedding = self.transformer.get_input_embeddings()
        hidden_size = self.transformer.config.hidden_size

        # Classification head: predicts P(correct)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        ).float()

        # Move entire module (transformer + classifier) to the resolved device
        self.device = resolved_device
        self.to(self.device)

    def forward_from_embeddings(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass from embeddings → P(correct)."""
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states.float() * mask_expanded).sum(dim=1)
        mean_hidden = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(mean_hidden)
        # Return P(correct) as a differentiable scalar
        return F.softmax(logits, dim=-1)[:, 1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        embeds = self.embedding(input_ids)
        return self.forward_from_embeddings(embeds, attention_mask)

    def forward_soft_sparse(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient forward from sparse soft tokens.
        topk_indices: [batch, seq, k]
        topk_weights: [batch, seq, k] — carries gradients from Gumbel-Softmax

        All inputs are moved to self.device before use to guarantee no
        cross-device mixing with the proxy's embedding/transformer.
        """
        topk_indices = topk_indices.to(self.device)
        topk_weights = topk_weights.to(self.device)
        attention_mask = attention_mask.to(self.device)
        selected_embeds = self.embedding(topk_indices)  # [batch, seq, k, hidden]
        weights = topk_weights.to(selected_embeds.dtype).unsqueeze(-1)
        soft_embeddings = (weights * selected_embeds).sum(dim=2)  # [batch, seq, hidden]
        return self.forward_from_embeddings(soft_embeddings, attention_mask)


def generate_proxy_training_data(
    policy_model: nn.Module,
    tokenizer,
    dataloader: DataLoader,
    config: RLVRConfig,
    max_samples: int = 5000,
) -> List[Dict]:
    """
    Generate (prompt+solution, correctness) pairs for proxy verifier training.

    Samples solutions from the current policy, verifies them with the exact
    verifier, and returns labeled data.
    """
    print("Generating proxy verifier training data...")
    policy_model.eval()
    samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating proxy data"):
            if len(samples) >= max_samples:
                break
            input_ids = batch["input_ids"].to(DEVICES.policy)
            attention_mask = batch["attention_mask"].to(DEVICES.policy)
            ground_truths = batch["ground_truth"]

            # Generate 2 solutions per prompt (reduced from 4 for speed)
            for _ in range(2):
                outputs = policy_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )

                for i in range(outputs.shape[0]):
                    full_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    gen_text = tokenizer.decode(
                        outputs[i, input_ids.shape[1]:], skip_special_tokens=True
                    )
                    predicted = extract_answer_gsm8k(gen_text)
                    correct = verify_answer(predicted, ground_truths[i])

                    samples.append({
                        "input_ids": outputs[i].cpu(),
                        "attention_mask": (outputs[i] != tokenizer.pad_token_id).long().cpu(),
                        "label": int(correct),
                    })

            if len(samples) >= max_samples:
                break

    policy_model.train()

    positive_rate = sum(s['label'] for s in samples) / len(samples) if samples else 0
    print(f"Generated {len(samples)} samples. "
          f"Accuracy distribution: {positive_rate:.1%} correct")

    # Balance the dataset: oversample positives if severely imbalanced
    # This prevents the proxy from learning a trivial "always predict wrong" classifier
    positive_samples = [s for s in samples if s['label'] == 1]
    negative_samples = [s for s in samples if s['label'] == 0]

    if len(positive_samples) > 0 and positive_rate < 0.15:
        # Oversample positives to reach ~20% positive rate
        target_positive = max(len(negative_samples) // 4, len(positive_samples))
        oversampled_positives = []
        while len(oversampled_positives) < target_positive:
            oversampled_positives.extend(positive_samples)
        oversampled_positives = oversampled_positives[:target_positive]
        samples = negative_samples + oversampled_positives
        new_rate = sum(s['label'] for s in samples) / len(samples)
        print(f"  Rebalanced: {len(positive_samples)} → {len(oversampled_positives)} "
              f"positives (rate: {positive_rate:.1%} → {new_rate:.1%})")
    elif len(positive_samples) == 0:
        print("  WARNING: No correct samples generated! Proxy training will be degenerate.")
        # BLOCK degenerate proxy training: return None to signal caller
        if positive_rate < config.proxy_min_positive_rate:
            print(f"  ⛔ BLOCKING proxy retrain: positive rate {positive_rate:.1%} "
                  f"< minimum {config.proxy_min_positive_rate:.1%}")
            print(f"  Keeping current proxy verifier weights.")
            return None

    return samples


def train_proxy_verifier(
    proxy: ProxyVerifier,
    training_data: List[Dict],
    config: RLVRConfig,
    num_epochs: int = 2,
) -> ProxyVerifier:
    """Train or retrain the proxy verifier on labeled (solution, correctness) pairs."""
    print(f"Training proxy verifier on {len(training_data)} samples...")

    # Freeze transformer, train only classifier
    for param in proxy.transformer.parameters():
        param.requires_grad = False
    for param in proxy.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(proxy.classifier.parameters(), lr=config.proxy_verifier_lr)
    proxy.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Simple batching
        indices = np.random.permutation(len(training_data))
        batch_size = 8

        pbar = tqdm(range(0, len(indices), batch_size),
                    desc=f"Proxy Epoch {epoch + 1}/{num_epochs}")

        for start in pbar:
            batch_idx = indices[start:start + batch_size]
            batch_items = [training_data[i] for i in batch_idx]

            # Pad to same length
            max_len = max(item["input_ids"].shape[0] for item in batch_items)
            input_ids = torch.zeros(len(batch_items), max_len, dtype=torch.long)
            attn_mask = torch.zeros(len(batch_items), max_len, dtype=torch.long)
            labels = torch.zeros(len(batch_items), dtype=torch.long)

            for i, item in enumerate(batch_items):
                seq_len = item["input_ids"].shape[0]
                input_ids[i, :seq_len] = item["input_ids"]
                attn_mask[i, :seq_len] = item["attention_mask"]
                labels[i] = item["label"]
            input_ids = input_ids.to(DEVICES.proxy)
            attn_mask = attn_mask.to(DEVICES.proxy)
            labels = labels.to(DEVICES.proxy)

            embeds = proxy.embedding(input_ids)
            outputs = proxy.transformer(
                inputs_embeds=embeds,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            mask_exp = attn_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            logits = proxy.classifier(pooled)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    print(f"Proxy verifier training complete. Final accuracy: {correct/total:.3f}")

    proxy.eval()
    for param in proxy.parameters():
        param.requires_grad = False
    return proxy


# ============================================================================
# PROXY REPLAY BUFFER (stable proxy training across distribution shifts)
# ============================================================================

class ProxyReplayBuffer:
    """
    Stores historical (solution, correctness) pairs for stable proxy training.

    Instead of generating fresh proxy data each time (slow + subject to collapse),
    this buffer maintains a mix of old high-quality samples and new samples.
    This prevents the catastrophic feedback loop where a collapsed policy generates
    all-wrong data → proxy learns trivial classifier → no gradient signal.
    """

    def __init__(self, max_size: int = 8000):
        self.buffer = []
        self.max_size = max_size

    def add(self, samples: List[Dict]):
        """Add new samples, keeping buffer capped."""
        if samples is None:
            return
        self.buffer.extend(samples)
        if len(self.buffer) > self.max_size:
            # Keep mix of old and new: remove oldest first
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, n: int) -> List[Dict]:
        """Sample from buffer for proxy training."""
        n = min(n, len(self.buffer))
        if n == 0:
            return []
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def positive_rate(self) -> float:
        """Fraction of correct samples in buffer."""
        if not self.buffer:
            return 0.0
        return sum(s['label'] for s in self.buffer) / len(self.buffer)

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DEGENERACY DETECTION AND PENALTY
# ============================================================================

def compute_degeneracy_penalty(
    hard_tokens: torch.Tensor,
    min_unique_ratio: float = 0.1,
    penalty_coef: float = 2.0,
) -> torch.Tensor:
    """
    Penalize outputs with extreme token repetition (e.g., '000000...' or '######000...').

    After model collapse, the policy generates degenerate outputs where the same
    token is repeated hundreds of times. This function detects that pattern and
    returns a scalar penalty to add to the loss.

    Args:
        hard_tokens: [batch, seq_len] — argmax token IDs from generation
        min_unique_ratio: Below this unique/total ratio → degenerate
        penalty_coef: Multiplier for the penalty

    Returns:
        Scalar penalty tensor (on same device as hard_tokens)
    """
    batch_size = hard_tokens.shape[0]
    penalties = []
    for i in range(batch_size):
        tokens = hard_tokens[i]
        unique_ratio = tokens.unique().numel() / max(tokens.numel(), 1)
        # Low unique ratio = degenerate (e.g., '000000...' has ratio ~0.004)
        if unique_ratio < min_unique_ratio:
            penalty = (min_unique_ratio - unique_ratio) * penalty_coef
        else:
            penalty = 0.0
        penalties.append(penalty)
    return torch.tensor(np.mean(penalties), device=hard_tokens.device, dtype=torch.float32)


# ============================================================================
# GRADE-STE TRAINER FOR RLVR
# ============================================================================

class GRADERLVRTrainer:
    """
    GRADE-STE trainer adapted for RLVR with math reasoning.

    Changes from the original GumbelTrainerMemoryEfficient:
    1. Uses ProxyVerifier instead of SameVocabRewardModel
    2. Logs exact verification accuracy alongside proxy reward
    3. Supports periodic proxy retraining to combat distribution shift
    4. Computes KL against reference model for regularization

    The training loop:
      1. Generate soft tokens via Gumbel-Softmax STE
      2. Compute differentiable proxy reward R̂(soft_tokens)
      3. Decode hard tokens → extract answer → exact verify (for logging)
      4. Loss = -R̂ + β * KL(π_θ || π_ref)
      5. Backpropagate through the full differentiable graph
    """

    def __init__(
        self,
        generator: DifferentiableGenerator,
        ref_model: nn.Module,
        proxy_verifier: ProxyVerifier,
        tokenizer,
        config: RLVRConfig,
    ):
        self.generator = generator
        self.ref_model = ref_model
        self.proxy_verifier = proxy_verifier
        self.tokenizer = tokenizer
        self.config = config
        self.topk = config.gumbel_topk if config.gumbel_topk > 0 else 256

        self.optimizer = torch.optim.AdamW(
            generator.model.parameters(), lr=config.learning_rate
        )
        self.step_count = 0
        # Reward grounding: track disagreement between proxy and exact verifier
        self._proxy_accuracy_ema = 0.0

    def _policy_forward_step(self, policy_embeds, policy_mask):
        outputs = self.generator.model(
            inputs_embeds=policy_embeds,
            attention_mask=policy_mask,
            use_cache=False,
        )
        return outputs.logits[:, -1, :]

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        ground_truths: List[str],
    ) -> dict:
        """
        Single GRADE-STE training step.

        Args:
            prompt_ids:    [batch, prompt_len]
            prompt_mask:   [batch, prompt_len]
            ground_truths: List of ground truth answer strings
        """
        tau = self.generator.get_tau(self.step_count)
        batch_size = prompt_ids.shape[0]
        device = DEVICES.policy

        torch.cuda.empty_cache()

        # ================================================================
        # PHASE 1: Soft generation with online KL
        # ================================================================
        policy_embeds = self.generator.embedding(prompt_ids)
        policy_mask = prompt_mask.clone()

        # Reference model KV-cache for efficient KL computation
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=prompt_ids.to(DEVICES.ref),
                attention_mask=prompt_mask.to(DEVICES.ref),
                use_cache=True,
            )
            ref_past_kv = ref_outputs.past_key_values
            del ref_outputs

        topk_indices_list = []
        topk_weights_list = []
        hard_tokens_list = []
        kl_sum = 0.0

        for step_idx in range(self.config.max_new_tokens):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                policy_logits = torch.utils.checkpoint.checkpoint(
                    self._policy_forward_step,
                    policy_embeds,
                    policy_mask,
                    use_reentrant=False,
                ).float()

            # Top-k Gumbel-Softmax with STE
            soft_token, topk_idx = gumbel_softmax_topk(
                policy_logits, tau=tau, k=self.topk, hard=True  # STE
            )

            topk_weights = soft_token.gather(-1, topk_idx)
            topk_indices_list.append(topk_idx.detach())
            topk_weights_list.append(topk_weights)

            hard_token = soft_token.argmax(dim=-1)
            hard_tokens_list.append(hard_token)

            # Reference model forward with KV-cache (no grad)
            with torch.no_grad():
                ref_attn = torch.cat([
                    policy_mask[:, :prompt_ids.shape[1] + step_idx].to(DEVICES.ref),
                    torch.ones(batch_size, 1, device=DEVICES.ref),
                ], dim=1)
                ref_out = self.ref_model(
                    input_ids=hard_token.unsqueeze(-1).to(DEVICES.ref),
                    attention_mask=ref_attn,
                    past_key_values=ref_past_kv,
                    use_cache=True,
                )
                ref_logits = ref_out.logits[:, -1, :].float().to(device)
                ref_past_kv = ref_out.past_key_values
                del ref_out

                # Online KL — computed WITH gradients to provide a differentiable
                # regularization signal, not just for logging
                ref_lp = F.log_softmax(ref_logits, dim=-1)
                del ref_logits

            # KL computed outside no_grad so it contributes gradients
            policy_lp = F.log_softmax(policy_logits, dim=-1)  # has grad
            policy_p = policy_lp.exp()
            kl_per_token = (policy_p * (policy_lp - ref_lp.to(device))).sum(dim=-1)
            kl_sum = kl_sum + kl_per_token.mean()  # Keep as tensor for grad
            del policy_lp, ref_lp

            # Update embeddings for next step
            next_embed = (
                soft_token.to(self.generator.embedding.weight.dtype)
                @ self.generator.embedding.weight
            ).unsqueeze(1)
            policy_embeds = torch.cat([policy_embeds, next_embed], dim=1)
            policy_mask = torch.cat([
                policy_mask, torch.ones(batch_size, 1, device=device)
            ], dim=1)

            del policy_logits, soft_token

        del ref_past_kv
        torch.cuda.empty_cache()

        # ================================================================
        # PHASE 2: Compute differentiable proxy reward
        # ================================================================
        topk_indices = torch.stack(topk_indices_list, dim=1)
        topk_weights = torch.stack(topk_weights_list, dim=1)
        hard_tokens = torch.stack(hard_tokens_list, dim=1)
        del topk_indices_list, topk_weights_list, hard_tokens_list

        gen_mask = torch.ones(batch_size, self.config.max_new_tokens, device=DEVICES.proxy)

        # Differentiable reward from proxy verifier (move tensors to proxy device)
        proxy_rewards = self.proxy_verifier.forward_soft_sparse(
            topk_indices.to(DEVICES.proxy), topk_weights.to(DEVICES.proxy), gen_mask
        )

        # ================================================================
        # PHASE 3: Exact verification for logging (no grad)
        # ================================================================
        with torch.no_grad():
            exact_rewards = []
            for i in range(batch_size):
                gen_text = self.tokenizer.decode(hard_tokens[i].cpu(), skip_special_tokens=True)
                predicted = extract_answer_gsm8k(gen_text)
                exact_reward = verify_answer(predicted, ground_truths[i])
                exact_rewards.append(exact_reward)
            exact_reward_mean = np.mean(exact_rewards)

        # ================================================================
        # PHASE 4: Loss and backprop
        # ================================================================
        kl_mean = kl_sum / self.config.max_new_tokens  # now a differentiable tensor
        proxy_reward_mean = proxy_rewards.mean().to(device)

        # Clamp proxy reward to prevent exploitation of overconfident proxy
        proxy_reward_clamped = torch.clamp(proxy_reward_mean, 0.0, 0.95)

        # --- REWARD GROUNDING: scale down proxy when it disagrees with exact ---
        disagreement = abs(proxy_reward_clamped.item() - exact_reward_mean)
        self._proxy_accuracy_ema = 0.95 * self._proxy_accuracy_ema + 0.05 * disagreement
        trust_factor = max(0.1, 1.0 - self._proxy_accuracy_ema)
        proxy_reward_grounded = proxy_reward_clamped * trust_factor

        # --- KL LOWER BOUND: penalize if KL drops too low (prevents collapse) ---
        kl_lower_penalty = torch.relu(
            torch.tensor(self.config.kl_lower_bound, device=device) - kl_mean
        ) * self.config.kl_lower_coef

        # --- DEGENERACY PENALTY: penalize repetitive outputs like '000000...' ---
        degen_penalty = compute_degeneracy_penalty(
            hard_tokens,
            min_unique_ratio=self.config.min_unique_token_ratio,
            penalty_coef=self.config.degeneracy_penalty_coef,
        )

        loss = (-proxy_reward_grounded
                + self.config.kl_coef * kl_mean
                + kl_lower_penalty
                + degen_penalty)

        self.optimizer.zero_grad()
        loss.backward()

        del topk_indices, topk_weights, policy_embeds, proxy_rewards
        torch.cuda.empty_cache()

        grad_norms = [
            p.grad.norm().item()
            for p in self.generator.model.parameters()
            if p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1

        kl_val = kl_mean.item() if torch.is_tensor(kl_mean) else kl_mean

        result = {
            "loss": loss.item(),
            "proxy_reward": proxy_reward_mean.item(),
            "proxy_reward_grounded": proxy_reward_grounded.item(),
            "exact_accuracy": exact_reward_mean,
            "kl": kl_val,
            "kl_lower_penalty": kl_lower_penalty.item(),
            "degen_penalty": degen_penalty.item(),
            "trust_factor": trust_factor,
            "tau": tau,
            "grad_norm_mean": np.mean(grad_norms) if grad_norms else 0,
            "grad_norm_std": np.std(grad_norms) if grad_norms else 0,
        }

        del loss, kl_mean, proxy_reward_mean
        torch.cuda.empty_cache()
        return result


# ============================================================================
# GRPO TRAINER FOR RLVR (baseline comparison)
# ============================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization with verifiable rewards.

    This is the standard RLVR baseline (DeepSeek-R1 style).
    For each prompt, sample G completions, verify each with exact match,
    compute group-normalized advantages, and apply clipped policy gradient.

    Included here for fair comparison against GRADE-STE.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        config: RLVRConfig,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
        self.step_count = 0

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        ground_truths: List[str],
    ) -> dict:
        """
        Single GRPO step.

        1. Sample G completions per prompt
        2. Verify each with exact match
        3. Compute group-normalized advantages
        4. Apply clipped policy gradient
        """
        batch_size = prompt_ids.shape[0]
        G = self.config.grpo_group_size
        device = DEVICES.policy

        # ================================================================
        # PHASE 1: Sample G completions per prompt (MICRO-BATCHED)
        # ================================================================
        # Expand prompts: [B, L] → [B*G, L]
        expanded_ids = prompt_ids.repeat_interleave(G, dim=0)
        expanded_mask = prompt_mask.repeat_interleave(G, dim=0)

        # Micro-batch generation to prevent CUDA OOM
        # (group_size=8 with batch_size=1 → 8 completions, process 4 at a time)
        micro_batch = self.config.grpo_micro_batch
        all_generated = []
        total_expanded = expanded_ids.shape[0]

        with torch.no_grad():
            for mb_start in range(0, total_expanded, micro_batch):
                mb_end = min(mb_start + micro_batch, total_expanded)
                mb_ids = expanded_ids[mb_start:mb_end]
                mb_mask = expanded_mask[mb_start:mb_end]
                mb_gen = self.policy.generate(
                    mb_ids,
                    attention_mask=mb_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    min_new_tokens=self.config.min_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                all_generated.append(mb_gen)
                torch.cuda.empty_cache()

        # Pad micro-batches to same length and concatenate
        max_gen_len = max(g.shape[1] for g in all_generated)
        padded = []
        for g in all_generated:
            if g.shape[1] < max_gen_len:
                pad = torch.full(
                    (g.shape[0], max_gen_len - g.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=g.dtype, device=g.device,
                )
                g = torch.cat([g, pad], dim=1)
            padded.append(g)
        generated = torch.cat(padded, dim=0)

        prompt_len = prompt_ids.shape[1]
        gen_tokens = generated[:, prompt_len:]  # [B*G, gen_len]
        gen_mask = (gen_tokens != self.tokenizer.pad_token_id).float()
        response_mask = (generated != self.tokenizer.pad_token_id).long()

        # ================================================================
        # PHASE 2: Exact verification → rewards
        # ================================================================
        rewards = torch.zeros(batch_size * G, device=device)
        for i in range(batch_size * G):
            gen_text = self.tokenizer.decode(gen_tokens[i], skip_special_tokens=True)
            predicted = extract_answer_gsm8k(gen_text)
            gt_idx = i // G
            rewards[i] = verify_answer(predicted, ground_truths[gt_idx])

        # ================================================================
        # PHASE 3: Group-normalized advantages
        # ================================================================
        rewards_grouped = rewards.view(batch_size, G)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards_grouped - group_mean) / group_std).view(-1)

        # Skip if no variance in rewards (all correct or all wrong)
        # But log number of groups with variance for diagnostics
        groups_with_signal = (rewards_grouped.std(dim=1) > 1e-8).sum().item()
        if rewards.std() < 1e-8:
            return {
                "loss": 0.0,
                "exact_accuracy": rewards.mean().item(),
                "kl": 0.0,
                "advantages_mean": 0.0,
                "groups_with_signal": 0,
                "total_groups": batch_size,
            }

        # ================================================================
        # PHASE 4: Clipped policy gradient
        # ================================================================
        with torch.no_grad():
            old_outputs = self.policy(generated, attention_mask=response_mask)
            old_logits = old_outputs.logits[:, prompt_len - 1:-1, :]
            old_lp = F.log_softmax(old_logits, dim=-1)
            old_token_lp = old_lp.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

            ref_outputs = self.ref_model(
                generated.to(DEVICES.ref), attention_mask=response_mask.to(DEVICES.ref)
            )
            ref_logits = ref_outputs.logits[:, prompt_len - 1:-1, :]
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_token_lp = ref_lp.gather(
                -1, gen_tokens.to(DEVICES.ref).unsqueeze(-1)
            ).squeeze(-1).to(device)

        # Forward with gradients
        outputs = self.policy(generated, attention_mask=response_mask)
        new_logits = outputs.logits[:, prompt_len - 1:-1, :]
        new_lp = F.log_softmax(new_logits, dim=-1)
        new_token_lp = new_lp.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

        # Clipped ratio
        log_ratio = new_token_lp - old_token_lp
        log_ratio = torch.clamp(log_ratio, -2.0, 2.0)
        ratio = torch.exp(log_ratio)

        # Per-sequence advantage (broadcast to all tokens)
        adv = advantages.unsqueeze(-1).expand_as(gen_mask)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.config.grpo_clip, 1 + self.config.grpo_clip) * adv
        policy_loss = -(torch.min(surr1, surr2) * gen_mask).sum() / gen_mask.sum().clamp(min=1)

        # KL penalty
        kl_per_token = new_token_lp - ref_token_lp
        kl = (kl_per_token * gen_mask).sum() / gen_mask.sum().clamp(min=1)

        loss = policy_loss + self.config.kl_coef * kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "exact_accuracy": rewards.mean().item(),
            "kl": kl.item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "groups_with_signal": groups_with_signal,
            "total_groups": batch_size,
        }


# ============================================================================
# HYBRID TRAINER: GRADE + GRPO
# ============================================================================

class HybridGRADEGRPOTrainer:
    """
    Combines GRADE-STE (differentiable proxy) with GRPO (exact verifier).

    Strategy:
      - GRADE provides low-variance gradient signal via differentiable proxy
      - GRPO provides unbiased signal via exact verification
      - Combined loss = α * L_GRADE + (1-α) * L_GRPO

    This approach gets the best of both worlds:
      - GRADE's stable gradients for smooth optimization
      - GRPO's exact rewards to prevent reward hacking of the proxy
    """

    def __init__(
        self,
        generator: DifferentiableGenerator,
        ref_model: nn.Module,
        proxy_verifier: ProxyVerifier,
        tokenizer,
        config: RLVRConfig,
    ):
        self.grade_trainer = GRADERLVRTrainer(
            generator, ref_model, proxy_verifier, tokenizer, config
        )
        self.grpo_trainer = GRPOTrainer(
            generator.model, ref_model, tokenizer, config
        )
        self.config = config
        self.generator = generator

    def step(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        ground_truths: List[str],
    ) -> dict:
        """Alternating or combined GRADE + GRPO steps."""
        # Simple alternating schedule: GRADE on even steps, GRPO on odd
        # (could also do weighted combination within a single step)
        step = self.grade_trainer.step_count

        if step % 2 == 0:
            # GRADE step
            result = self.grade_trainer.step(prompt_ids, prompt_mask, ground_truths)
            result["method"] = "grade"
        else:
            # GRPO step
            result = self.grpo_trainer.step(prompt_ids, prompt_mask, ground_truths)
            result["method"] = "grpo"
            self.grade_trainer.step_count += 1  # Keep step counts synced

        return result


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    dataloader: DataLoader,
    config: RLVRConfig,
    max_batches: int = 50,
    desc: str = "Evaluating",
) -> Dict:
    """
    Evaluate model on math problems with exact verification.

    Returns accuracy, average answer extraction rate, and sample outputs.
    """
    model.eval()
    correct = 0
    total = 0
    extracted = 0
    sample_outputs = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(DEVICES.policy)
        attention_mask = batch["attention_mask"].to(DEVICES.policy)
        ground_truths = batch["ground_truth"]

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,  # Greedy for eval
            pad_token_id=tokenizer.pad_token_id,
        )

        for i in range(outputs.shape[0]):
            gen_text = tokenizer.decode(
                outputs[i, input_ids.shape[1]:], skip_special_tokens=True
            )
            predicted = extract_answer_gsm8k(gen_text)

            if predicted is not None:
                extracted += 1

            is_correct = verify_answer(predicted, ground_truths[i])
            correct += is_correct
            total += 1

            if len(sample_outputs) < 5:
                sample_outputs.append({
                    "question": batch["question"][i][:100],
                    "ground_truth": ground_truths[i],
                    "predicted": predicted,
                    "correct": bool(is_correct),
                    "generation": gen_text[:200],
                })

    model.train()

    accuracy = correct / total if total > 0 else 0
    extraction_rate = extracted / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "extraction_rate": extraction_rate,
        "total": total,
        "correct": correct,
        "samples": sample_outputs,
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def setup_model_and_tokenizer(config: RLVRConfig):
    """Initialize model, tokenizer, LoRA, reference model."""
    print(f"Loading model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # CRITICAL FIX: Decoder-only models MUST use left-padding.
    # Right-padding causes the model to attend to pad tokens during generation,
    # producing degraded and often nonsensical outputs.
    tokenizer.padding_side = 'left'
    print(f"  Tokenizer padding_side set to: '{tokenizer.padding_side}'")

    # Policy model → DEVICES.policy
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    ).to(DEVICES.policy)

    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Reference model → DEVICES.ref (can be a different GPU)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=None,
    ).to(DEVICES.ref)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return model, ref_model, tokenizer


def main(config: Optional[RLVRConfig] = None):
    """Main training entry point."""
    if config is None:
        config = RLVRConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    config.output_dir = f"{config.output_dir}/{config.training_mode}"

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(f"{config.output_dir}/config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)

    # Setup
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)
    data_splits = RLVRDataSplits(config, tokenizer)

    # Initialize wandb
    wandb.init(
        project="grade-rlvr-gsm8k",
        config=vars(config),
        name=f"grade-rlvr-{config.training_mode}-{config.base_model.split('/')[-1]}",
    )

    # ================================================================
    # Phase 1: Train proxy verifier
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training proxy verifier")
    print("=" * 60)

    proxy_path = Path(config.output_dir) / "proxy_verifier.pt"
    proxy_verifier = ProxyVerifier(config.base_model, str(DEVICES.proxy))

    if proxy_path.exists():
        print(f"Loading proxy verifier from {proxy_path}")
        proxy_verifier.classifier.load_state_dict(
            torch.load(proxy_path, map_location=DEVICES.proxy)
        )
        proxy_verifier.eval()
        for param in proxy_verifier.parameters():
            param.requires_grad = False
    else:
        # Generate training data from base model
        proxy_dataloader = data_splits.get_proxy_train_dataloader(batch_size=4)
        proxy_data = generate_proxy_training_data(
            model, tokenizer, proxy_dataloader, config,
            max_samples=config.proxy_train_samples,
        )
        proxy_verifier = train_proxy_verifier(proxy_verifier, proxy_data, config)
        torch.save(proxy_verifier.classifier.state_dict(), proxy_path)

    # ================================================================
    # Phase 2: Policy training
    # ================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 2: Policy training ({config.training_mode})")
    print("=" * 60)

    generator = DifferentiableGenerator(model, tokenizer, config)

    if config.training_mode == "grade_only":
        trainer = GRADERLVRTrainer(
            generator, ref_model, proxy_verifier, tokenizer, config
        )
    elif config.training_mode == "grpo_only":
        trainer = GRPOTrainer(model, ref_model, tokenizer, config)
    elif config.training_mode == "hybrid":
        trainer = HybridGRADEGRPOTrainer(
            generator, ref_model, proxy_verifier, tokenizer, config
        )
    else:
        raise ValueError(f"Unknown training_mode: {config.training_mode}")

    train_dataloader = data_splits.get_policy_train_dataloader(config.batch_size)
    val_dataloader = data_splits.get_val_dataloader(batch_size=4)

    # Training loop
    # all_metrics = defaultdict(list)
    # train_iter = iter(train_dataloader)
    # best_val_accuracy = 0.0
    # Training loop
    all_metrics = defaultdict(list)
    train_iter = iter(train_dataloader)
    best_val_accuracy = 0.0
    collapse_counter = 0  # Track consecutive eval failures for early stopping
    recovery_attempts = 0  # Track how many times we've tried to recover

    # Initialize replay buffer for stable proxy training
    replay_buffer = ProxyReplayBuffer(max_size=config.proxy_replay_buffer_size)

    # Seed replay buffer with initial proxy training data (if available)
    if config.training_mode in ('grade_only', 'hybrid'):
        print("Seeding replay buffer with initial proxy data...")
        initial_proxy_dl = data_splits.get_proxy_train_dataloader(
            batch_size=config.proxy_data_batch_size
        )
        initial_proxy_data = generate_proxy_training_data(
            model, tokenizer, initial_proxy_dl, config,
            max_samples=config.proxy_train_samples // 4,
        )
        if initial_proxy_data is not None:
            replay_buffer.add(initial_proxy_data)
            print(f"  Replay buffer seeded with {len(replay_buffer)} samples "
                  f"(positive rate: {replay_buffer.positive_rate():.1%})")

    cuda_error_count = 0
    max_cuda_errors = 5  # Stop after this many CUDA errors

    for step in tqdm(range(config.max_steps), desc="Training"):
        # Get batch (cycle through data)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        # ✅ Correct device handling — tensors go to policy device
        prompt_ids = batch["input_ids"].to(DEVICES.policy)
        prompt_mask = batch["attention_mask"].to(DEVICES.policy)
        ground_truths = batch["ground_truth"]

        # ============================================================
        # Training step (with CUDA error recovery)
        # ============================================================
        try:
            metrics = trainer.step(prompt_ids, prompt_mask, ground_truths)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                cuda_error_count += 1
                print(f"\n⚠ CUDA error at step {step} ({cuda_error_count}/{max_cuda_errors}): {e}")
                if cuda_error_count >= max_cuda_errors:
                    print(f"🛑 Too many CUDA errors ({cuda_error_count}). Stopping training.")
                    break
                # Try to recover: clear CUDA cache and skip this step
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            else:
                raise  # Re-raise non-CUDA errors

        # ============================================================
        # ✅ FIX: Log metrics EVERY step (important)
        # ============================================================
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                all_metrics[k].append(v)
        # Alias 'exact_accuracy' as 'reward' for analysis_script.py compatibility
        if 'exact_accuracy' in metrics:
            all_metrics['reward'].append(metrics['exact_accuracy'])

        # ============================================================
        # WandB logging
        # ============================================================
        log_dict = {
            f"train/{k}": v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        log_dict["train/step"] = step
        wandb.log(log_dict, step=step)

        # ============================================================
        # Print progress
        # ============================================================
        if step % 50 == 0:
            metric_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))
            )
            print(f"Step {step}: {metric_str}")

        # ============================================================
        # Evaluation
        # ============================================================
        if step > 0 and step % config.eval_every == 0:
            print(f"\n--- Evaluation at step {step} ---")

            eval_results = evaluate(
                model, tokenizer, val_dataloader, config,
                max_batches=25, desc="Validation",
            )

            print(f"  Val Accuracy:      {eval_results['accuracy']:.4f}")
            print(f"  Extraction Rate:   {eval_results['extraction_rate']:.4f}")
            print(f"  Correct/Total:     {eval_results['correct']}/{eval_results['total']}")

            # Sample outputs
            for sample in eval_results["samples"][:2]:
                print(f"  Q: {sample['question']}")
                print(f"  GT: {sample['ground_truth']} | Pred: {sample['predicted']} "
                    f"| {'✓' if sample['correct'] else '✗'}")
                print(f"  Gen: {sample['generation'][:100]}...")
                print()

            wandb.log({
                "val/accuracy": eval_results["accuracy"],
                "val/extraction_rate": eval_results["extraction_rate"],
            }, step=step)

            # Track validation metrics for analysis_script.py
            all_metrics['val_reward'].append(eval_results['accuracy'])

            # Save best model
            if eval_results["accuracy"] > best_val_accuracy:
                best_val_accuracy = eval_results["accuracy"]
                all_metrics['best_val_reward'] = best_val_accuracy
                all_metrics['best_val_step'] = step
                collapse_counter = 0  # Reset collapse counter on improvement
                save_path = Path(config.output_dir) / "best_model"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  New best model saved (accuracy: {best_val_accuracy:.4f})")

            # ============================================================
            # Early stopping: detect model collapse
            # ============================================================
            recent_kl = all_metrics.get('kl', [0])[-1]
            is_collapsing = False

            if (config.training_mode in ('grade_only', 'hybrid') and
                eval_results['extraction_rate'] < config.extraction_rate_threshold):
                print(f"  ⚠ WARNING: Extraction rate {eval_results['extraction_rate']:.2f} "
                      f"< threshold {config.extraction_rate_threshold}")
                is_collapsing = True

            if (config.training_mode in ('grade_only', 'hybrid') and
                recent_kl < config.kl_collapse_threshold and step > config.warmup_steps):
                print(f"  ⚠ WARNING: KL divergence {recent_kl:.4f} "
                      f"< threshold {config.kl_collapse_threshold} — possible distribution collapse")
                is_collapsing = True

            if is_collapsing:
                collapse_counter += 1
                print(f"  Collapse counter: {collapse_counter}/{config.collapse_patience}")
                if collapse_counter >= config.collapse_patience:
                    recovery_attempts += 1
                    if recovery_attempts > config.max_recovery_attempts:
                        print(f"\n🛑 FINAL STOP: Model collapse persists after "
                              f"{recovery_attempts - 1} recovery attempts.")
                        print(f"   Restoring best model (accuracy: {best_val_accuracy:.4f})")
                        best_path = Path(config.output_dir) / "best_model"
                        if best_path.exists():
                            from peft import PeftModel
                            model.load_adapter(str(best_path), adapter_name="default")
                            print(f"   Best model restored from {best_path}")
                        break

                    print(f"\n⚠️ RECOVERY ATTEMPT {recovery_attempts}/{config.max_recovery_attempts}: "
                          f"Model collapse detected.")

                    # 1. Restore best model weights
                    best_path = Path(config.output_dir) / "best_model"
                    if best_path.exists():
                        from peft import PeftModel
                        model.load_adapter(str(best_path), adapter_name="default")
                        print(f"   ✅ Restored best model from {best_path}")

                    # 2. Reset optimizer state (stale momentum causes instability)
                    reduced_lr = config.learning_rate * (0.5 ** recovery_attempts)
                    if hasattr(trainer, 'optimizer'):
                        trainer.optimizer = torch.optim.AdamW(
                            model.parameters(), lr=reduced_lr
                        )
                        print(f"   ✅ Optimizer reset with reduced LR: {reduced_lr:.2e}")

                    # 3. Retrain proxy from replay buffer if quality is sufficient
                    if (config.training_mode in ('grade_only', 'hybrid') and
                        len(replay_buffer) > 0 and
                        replay_buffer.positive_rate() >= config.proxy_min_positive_rate):
                        print(f"   🔄 Retraining proxy from replay buffer "
                              f"({len(replay_buffer)} samples, "
                              f"{replay_buffer.positive_rate():.1%} positive)")
                        buffer_data = replay_buffer.sample(config.proxy_train_samples)
                        for param in proxy_verifier.classifier.parameters():
                            param.requires_grad = True
                        proxy_verifier = train_proxy_verifier(
                            proxy_verifier, buffer_data, config, num_epochs=2
                        )
                        print(f"   ✅ Proxy retrained from replay buffer")
                    else:
                        print(f"   ⚠️ Replay buffer too low quality for proxy retrain "
                              f"(positive rate: {replay_buffer.positive_rate():.1%})")

                    collapse_counter = 0  # Reset and continue training
                    print(f"   Continuing training with recovery settings...")
            else:
                collapse_counter = 0

        # ============================================================
        # Proxy verifier retraining (with quality gate + replay buffer)
        # ============================================================
        if (config.training_mode in ("grade_only", "hybrid") and
            step > 0 and step % config.proxy_verifier_retrain_every == 0):

            print(f"\n--- Retraining proxy verifier at step {step} ---")

            proxy_dataloader = data_splits.get_proxy_train_dataloader(
                batch_size=config.proxy_data_batch_size
            )

            proxy_data = generate_proxy_training_data(
                model, tokenizer, proxy_dataloader, config,
                max_samples=config.proxy_train_samples // 2,
            )

            # Add new data to replay buffer (even if None, buffer handles it)
            replay_buffer.add(proxy_data)
            print(f"  Replay buffer: {len(replay_buffer)} samples "
                  f"(positive rate: {replay_buffer.positive_rate():.1%})")

            # Quality gate: only retrain if we have good data
            if proxy_data is None:
                print(f"  ⛔ Skipping proxy retrain (blocked by quality gate)")
                # Try training from replay buffer instead
                if (len(replay_buffer) >= config.proxy_train_samples // 4 and
                    replay_buffer.positive_rate() >= config.proxy_min_positive_rate):
                    print(f"  🔄 Using replay buffer data instead")
                    proxy_data = replay_buffer.sample(config.proxy_train_samples // 2)
                else:
                    print(f"  ⚠️ Replay buffer also insufficient. Skipping.")
                    continue

            # Mix fresh data with replay buffer data for stability
            if len(replay_buffer) >= config.proxy_train_samples // 4:
                buffer_samples = replay_buffer.sample(config.proxy_train_samples // 4)
                proxy_data = proxy_data + buffer_samples
                print(f"  Mixed {len(proxy_data)} total samples "
                      f"(fresh + replay buffer)")

            # Unfreeze classifier
            for param in proxy_verifier.classifier.parameters():
                param.requires_grad = True

            proxy_verifier = train_proxy_verifier(
                proxy_verifier, proxy_data, config, num_epochs=1
            )

            torch.save(
                proxy_verifier.classifier.state_dict(),
                Path(config.output_dir) / "proxy_verifier.pt",
            )

    # ============================================================
    # Save results for analysis_script.py
    # ============================================================
    # Convert defaultdict to regular dict so json.dump works cleanly,
    # and ensure scalar 'best_val_*' fields are preserved (not inside lists)
    save_metrics = dict(all_metrics)
    with open(f"{config.output_dir}/results.json", "w") as f:
        json.dump(save_metrics, f)

    # ================================================================
    # Phase 3: Final test evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Final test evaluation")
    print("=" * 60)

    test_dataloader = data_splits.get_test_dataloader(batch_size=4)
    test_results = evaluate(
        model, tokenizer, test_dataloader, config,
        max_batches=100, desc="Test",
    )

    print(f"\n{'=' * 40}")
    print(f"FINAL TEST RESULTS ({config.training_mode})")
    print(f"{'=' * 40}")
    print(f"  Accuracy:        {test_results['accuracy']:.4f}")
    print(f"  Extraction Rate: {test_results['extraction_rate']:.4f}")
    print(f"  Correct/Total:   {test_results['correct']}/{test_results['total']}")

    wandb.log({
        "test/accuracy": test_results["accuracy"],
        "test/extraction_rate": test_results["extraction_rate"],
    })

    # Save final results
    with open(f"{config.output_dir}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Update results.json with test_eval so analysis_script.py can read it
    results_json_path = f"{config.output_dir}/results.json"
    try:
        with open(results_json_path, "r") as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    saved['test_eval'] = {
        'mean_reward': test_results['accuracy'],
        'std_reward': 0.0,  # single-run, no std across seeds
    }
    with open(results_json_path, "w") as f:
        json.dump(saved, f, indent=2)

    wandb.finish()
    return test_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRADE-STE for RLVR on GSM8K")
    parser.add_argument("--mode", type=str, default="grade_only",
                        choices=["grade_only", "grpo_only", "hybrid"],
                        help="Training mode")
    parser.add_argument("--model", type=str, default="/home/cccp/25m2118/RND2/Qwen2.5-3B-Instruct",
                        help="Base model")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--grade_weight", type=float, default=0.5)
    parser.add_argument("--grpo_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results_rlvr")

    args = parser.parse_args()

    config = RLVRConfig(
        training_mode=args.mode,
        base_model=args.model,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        grade_weight=args.grade_weight,
        grpo_weight=args.grpo_weight,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    main(config)
