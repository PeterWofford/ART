from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os

from dotenv import load_dotenv

import art
from art.megatron import MegatronBackend
from dev.yes_no_maybe.prompts import build_prompt_variants, slice_prompts
from dev.yes_no_maybe.runtime_config import (
    RunConfig,
    resolve_engine_args,
    resolve_run_config,
)
from dev.yes_no_maybe.trainer import run_training


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    prompt_offset: int
    prompt_limit: int
    target_reward: float
    max_steps: int


def default_curriculum(base_config: RunConfig) -> list[CurriculumStage]:
    target = base_config.target_reward
    steps = base_config.max_steps
    return [
        CurriculumStage("slice-2", prompt_offset=4, prompt_limit=2, target_reward=target, max_steps=steps),
        CurriculumStage("first-8", prompt_offset=0, prompt_limit=8, target_reward=target, max_steps=steps),
        CurriculumStage("first-24", prompt_offset=0, prompt_limit=24, target_reward=target, max_steps=steps),
        CurriculumStage("full-48", prompt_offset=0, prompt_limit=0, target_reward=target, max_steps=steps),
    ]


async def run_curriculum() -> None:
    load_dotenv()

    backend = MegatronBackend()
    try:
        engine_args = resolve_engine_args()
        base_model = "Qwen/Qwen3.5-35B-A3B"
        model = art.TrainableModel(
            name=os.environ.get("MODEL_NAME", "megatron-001"),
            project="yes-no-maybe-megatron",
            base_model=base_model,
            _internal_config=art.dev.InternalModelConfig(
                engine_args=engine_args,
            ),
        )
        await model.register(backend)

        base_config = resolve_run_config()
        for stage in default_curriculum(base_config):
            print(f"\n### Curriculum stage: {stage.name} ###")
            stage_config = replace(
                base_config,
                prompt_offset=stage.prompt_offset,
                prompt_limit=stage.prompt_limit,
                target_reward=stage.target_reward,
                max_steps=stage.max_steps,
            )
            prompts = slice_prompts(
                build_prompt_variants(),
                offset=stage_config.prompt_offset,
                limit=stage_config.prompt_limit,
            )
            await run_training(model, prompts, stage_config)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(run_curriculum())
