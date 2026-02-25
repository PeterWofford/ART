from __future__ import annotations

import asyncio

import art

from .rollout import rollout
from .runtime_config import RunConfig


def _iter_prompt_batches(prompts: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0 or batch_size >= len(prompts):
        return [prompts]
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


async def run_training(
    model: art.TrainableModel,
    prompts: list[str],
    run_config: RunConfig,
) -> None:
    openai_client = model.openai_client()
    semaphore = asyncio.Semaphore(run_config.rollout_concurrency)
    start_step = await model.get_step()

    prompt_batches = _iter_prompt_batches(prompts, run_config.prompt_batch_size)
    for step in range(start_step, start_step + run_config.max_steps):
        print(f"\n=== Step {step + 1} ===")
        total_reward = 0.0
        total_maybe = 0.0
        total_rollouts = 0
        for batch_prompts in prompt_batches:
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(
                            openai_client,
                            model.get_inference_name(),
                            prompt,
                            semaphore=semaphore,
                            max_tokens=run_config.rollout_max_tokens,
                            timeout_s=run_config.rollout_timeout_s,
                            temperature=run_config.rollout_temperature,
                        )
                        for _ in range(run_config.rollouts_per_prompt)
                    )
                    for prompt in batch_prompts
                )
            )
            trajectories = [t for group in train_groups for t in group.trajectories]
            total_reward += sum(trajectory.reward for trajectory in trajectories)
            total_maybe += sum(
                trajectory.metrics.get("label_maybe", 0.0)
                for trajectory in trajectories
            )
            total_rollouts += len(trajectories)
            if trajectories:
                await model.train(
                    train_groups,
                    config=art.TrainConfig(learning_rate=run_config.learning_rate),
                )
        avg_reward = total_reward / total_rollouts if total_rollouts else 0.0
        maybe_rate = total_maybe / total_rollouts if total_rollouts else 0.0
        print(
            f"Step {step + 1}: avg_reward={avg_reward:.3f} "
            f"maybe_rate={maybe_rate:.3f} "
            f"rollouts={total_rollouts}"
        )
        if avg_reward >= run_config.target_reward:
            print(
                f"Reached target reward {run_config.target_reward:.3f}; "
                "stopping before additional training."
            )
            break
