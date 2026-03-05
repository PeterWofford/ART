import asyncio
import json
import os
from pathlib import Path
import time

import art


async def main() -> None:
    project = os.environ.get("ART_METRICS_PROJECT", "metrics-taxonomy-smoke")
    model_name = os.environ.get(
        "ART_METRICS_MODEL", f"metrics-smoke-{int(time.time())}"
    )
    base_path = os.environ.get("ART_METRICS_BASE_PATH", ".art")

    model = art.Model(
        name=model_name,
        project=project,
        base_path=base_path,
        report_metrics=["wandb"],
    )

    for step in (1, 2):
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
