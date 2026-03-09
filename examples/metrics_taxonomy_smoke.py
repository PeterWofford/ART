import asyncio
import json
import os
from pathlib import Path
import time

import art
from art.metrics import track_api_cost


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage = _Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


@track_api_cost(
    source="llm_judge/decorator_demo",
    provider="openai",
    model_name="openai/gpt-oss-20b",
)
async def _mock_judge_call(step: int) -> _Response:
    return _Response(
        prompt_tokens=50 * step,
        completion_tokens=20 * step,
    )


async def main() -> None:
    project_spec = os.environ.get("ART_METRICS_PROJECT", "metrics-taxonomy-smoke")
    entity = os.environ.get("ART_METRICS_ENTITY")
    project = project_spec
    if entity is None and "/" in project_spec:
        split_entity, split_project = project_spec.split("/", 1)
        if split_entity and split_project:
            entity = split_entity
            project = split_project

    model_name = os.environ.get(
        "ART_METRICS_MODEL", f"metrics-smoke-{int(time.time())}"
    )
    base_path = os.environ.get("ART_METRICS_BASE_PATH", ".art")

    model = art.Model(
        name=model_name,
        project=project,
        entity=entity,
        base_path=base_path,
        report_metrics=["wandb"],
    )

    for step in (1, 2):
        train_token = model.activate_metrics_context("train")
        try:
            await _mock_judge_call(step)
        finally:
            train_token.var.reset(train_token)

        trajectories = [
            art.TrajectoryGroup(
                trajectories=[
                    art.Trajectory(
                        reward=0.4 + 0.1 * step,
                        metrics={
                            "judge_quality": 0.7 + 0.05 * step,
                            "reward/custom_prefixed": 0.2 * step,
                        },
                        messages_and_choices=[
                            {"role": "user", "content": f"smoke step {step}"},
                            {"role": "assistant", "content": "ok"},
                        ],
                    )
                ],
                exceptions=[],
            )
        ]

        await model.log(
            trajectories,
            split="train",
            step=step,
            metrics={
                "loss/train": 1.0 / step,
                "loss/grad_norm": 0.5 + 0.1 * step,
                "throughput/train_tok_per_sec": 1000.0 + 100.0 * step,
                "time/step_wall_s": 1.5 + 0.2 * step,
                "data/step_num_scenarios": 2.0,
                "data/step_actor_tokens": 120.0 + 10.0 * step,
                "costs_prefill": 0.10 * step,
                "costs_sample": 0.05 * step,
                "costs/train/llm_judge/correctness": 0.02 * step,
            },
        )

    history_path = Path(base_path) / project / "models" / model_name / "history.jsonl"
    print(f"Wrote history: {history_path}")

    with open(history_path) as f:
        rows = [json.loads(line) for line in f]

    print("\nLast row key excerpts:")
    last = rows[-1]
    show_prefixes = ("reward/", "loss/", "throughput/", "time/", "data/", "costs/")
    for key in sorted(last):
        if key.startswith(show_prefixes):
            print(f"{key}: {last[key]}")

    print("\nIf WANDB_API_KEY is set, metrics are also logged to W&B.")


if __name__ == "__main__":
    asyncio.run(main())
