"""Yes-no-maybe metrics demo for the LocalBackend `model.train()` path.

This keeps the same prompt family, rollout structure, and reward ordering as
`dev/yes-no-maybe.py` while adding explicit metrics taxonomy instrumentation for
actor/eval timing and data metrics, while relying on LocalBackend for automatic
step wall time and GPU cost logging.
"""

from __future__ import annotations

import asyncio
from itertools import permutations
import os
import time

from dotenv import load_dotenv
import openai

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import art
from art.local import LocalBackend


async def create_chat_completion(
    client: openai.AsyncOpenAI,
    *,
    model_name: str,
    messages: art.Messages,
    max_tokens: int,
    timeout: float,
) -> openai.types.chat.chat_completion.ChatCompletion:
    return await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def with_quotes(word: str) -> str:
    return f"'{word}'"


def build_prompts() -> list[str]:
    return [
        f"{prefix} with {', '.join([with_quotes(word) if use_quotes else word for word in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(permutation)
            for length in [3, 2]
            for permutation in permutations(["yes", "no", "maybe"], length)
        )
    ]


def reward_for_answer(content: str | None) -> float:
    if content == "yes":
        return 0.5
    if content == "no":
        return 0.75
    if content == "maybe":
        return 1.0
    return 0.0


def scenario_id_for_prompt(prompt: str) -> str:
    return prompt.replace(" ", "_").replace("'", "")


def response_total_tokens(
    response: openai.types.chat.chat_completion.ChatCompletion,
) -> int:
    usage = response.usage
    if usage is None:
        return 0
    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(usage.completion_tokens or 0)
    return prompt_tokens + completion_tokens


def total_actor_tokens(groups: list[art.TrajectoryGroup]) -> int:
    return sum(
        int(trajectory.metadata.get("actor_total_tokens", 0) or 0)
        for group in groups
        for trajectory in group.trajectories
    )


async def rollout(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await create_chat_completion(
        client,
        model_name=model.get_inference_name(),
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward_for_answer(content),
        metadata={
            "scenario_id": scenario_id_for_prompt(prompt),
            "actor_total_tokens": response_total_tokens(chat_completion),
        },
        metrics={
            "valid_answer": reward_for_answer(content) > 0.0,
        },
    )


async def evaluate(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompts: list[str],
    *,
    max_tokens: int,
    timeout: float,
) -> list[art.TrajectoryGroup]:
    groups = await art.gather_trajectory_groups(
        art.TrajectoryGroup(
            [
                rollout(
                    client,
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            ],
            metadata={"scenario_id": scenario_id_for_prompt(prompt)},
        )
        for prompt in prompts
    )
    return groups


def print_history_summary(model: art.TrainableModel) -> None:
    history_path = (
        model.base_path + f"/{model.project}/models/{model.name}/history.jsonl"
    )
    print(f"History: {history_path}")


def build_internal_config() -> art.dev.InternalModelConfig:
    return art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=float(
                os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
            ),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
        )
    )


async def main() -> None:
    load_dotenv()

    backend = LocalBackend()
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    project = os.environ.get("PROJECT", "yes-no-maybe-metrics")
    model = art.TrainableModel(
        name=os.environ.get("MODEL_NAME", f"yes-no-maybe-metrics-{int(time.time())}"),
        project=project,
        base_model=base_model,
        report_metrics=["wandb"],
        _internal_config=build_internal_config(),
    )
    try:
        await model.register(backend)

        prompts = build_prompts()
        eval_prompts = prompts[: int(os.environ.get("EVAL_PROMPTS", "12"))]
        openai_client = model.openai_client()
        max_steps = int(os.environ.get("NUM_STEPS", "20"))
        rollouts_per_prompt = int(os.environ.get("ROLLOUTS_PER_PROMPT", "32"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
        timeout = float(os.environ.get("TIMEOUT", "100"))
        eval_every_n_steps = int(os.environ.get("EVAL_EVERY_N_STEPS", "1"))
        learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))

        start_step = await model.get_step()
        for offset in range(max_steps):
            current_step = start_step + offset

            if (
                eval_every_n_steps > 0
                and (current_step - start_step) % eval_every_n_steps == 0
            ):
                eval_builder = model.metrics_builder("eval")
                with eval_builder.activate_context():
                    with eval_builder.measure("time/step_eval_s"):
                        val_groups = await evaluate(
                            openai_client,
                            model,
                            eval_prompts,
                            max_tokens=max_tokens,
                            timeout=timeout,
                        )
                    eval_builder.add_data(
                        step_actor_tokens=total_actor_tokens(val_groups)
                    )
                await model.log(val_groups, split="val", step=current_step)

            train_builder = model.metrics_builder("train")
            with train_builder.activate_context():
                with train_builder.measure("time/step_actor_s"):
                    train_groups = await art.gather_trajectory_groups(
                        (
                            art.TrajectoryGroup(
                                rollout(
                                    openai_client,
                                    model,
                                    prompt,
                                    max_tokens=max_tokens,
                                    timeout=timeout,
                                )
                                for _ in range(rollouts_per_prompt)
                            )
                            for prompt in prompts
                        )
                    )
                train_builder.add_data(
                    step_actor_tokens=total_actor_tokens(train_groups)
                )
                result = await backend.train(
                    model,
                    train_groups,
                    learning_rate=learning_rate,
                )

            await model.log(
                split="train",
                step=result.step,
                trajectories=train_groups,
                metrics=result.metrics,
            )
            print(f"step {result.step} complete")

        print_history_summary(model)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
