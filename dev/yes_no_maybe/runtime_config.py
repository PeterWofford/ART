from __future__ import annotations

from dataclasses import dataclass
import os

import torch

import art


@dataclass(frozen=True)
class RunConfig:
    prompt_offset: int
    prompt_limit: int
    prompt_batch_size: int
    max_steps: int
    rollouts_per_prompt: int
    learning_rate: float
    target_reward: float
    rollout_timeout_s: float
    rollout_temperature: float
    rollout_max_tokens: int
    rollout_concurrency: int


def _env_float(names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name in os.environ:
            return float(os.environ[name])
    return default


def resolve_engine_args() -> art.dev.EngineArgs:
    return {
        "gpu_memory_utilization": _env_float(
            ("ART_VLLM_GPU_MEMORY_UTILIZATION", "VLLM_GPU_MEMORY_UTILIZATION"), 0.8
        ),
        "tensor_parallel_size": torch.cuda.device_count(),
        "language_model_only": os.environ.get("VLLM_LANGUAGE_MODEL_ONLY", "1") == "1",
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "1") == "1",
        "max_model_len": int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768")),
    }


def resolve_run_config() -> RunConfig:
    return RunConfig(
        prompt_offset=int(os.environ.get("PROMPT_OFFSET", "0")),
        prompt_limit=int(os.environ.get("PROMPT_LIMIT", "0")),
        prompt_batch_size=int(os.environ.get("PROMPT_BATCH_SIZE", "0")),
        max_steps=int(os.environ.get("NUM_STEPS", "20")),
        rollouts_per_prompt=int(os.environ.get("ROLLOUTS_PER_PROMPT", "8")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "1e-4")),
        target_reward=float(os.environ.get("TARGET_REWARD", "0.95")),
        rollout_timeout_s=float(os.environ.get("ROLLOUT_TIMEOUT_S", "45")),
        rollout_temperature=float(os.environ.get("ROLLOUT_TEMPERATURE", "0.7")),
        rollout_max_tokens=int(os.environ.get("ROLLOUT_MAX_TOKENS", "6")),
        rollout_concurrency=int(os.environ.get("ROLLOUT_CONCURRENCY", "256")),
    )
