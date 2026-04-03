"""
train_ruler_pipeline.py

Experimental IVR nav RL entrypoint using PipelineTrainer + LocalBackend(dedicated).

This is a pilot path alongside dev/train_ruler.py. It keeps the current launcher,
validation, and dataset files, but uses PipelineTrainer for scenario progression
and resume semantics.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import art
import httpx
import litellm
from art.local import LocalBackend
from art.metrics import track_api_cost
from art.pipeline_trainer import PipelineTrainer
from art.pipeline_trainer.ivr_nav_pilot import (
    IVRNavPilotConfig,
    row_session_id,
    select_test_rows,
    split_train_holdout_by_session,
)
from art.rewards import ruler_score_group
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from tqdm.asyncio import tqdm

load_dotenv()
os.environ.setdefault("WANDB_ENTITY", "coreweave1")

litellm.model_cost["openai/gpt-5.4"] = {
    "input_cost_per_token": 2.50 / 1_000_000,
    "cache_read_input_token_cost": 0.25 / 1_000_000,
    "output_cost_per_token": 15.00 / 1_000_000,
    "litellm_provider": "openai",
}
litellm.model_cost["openai/gpt-5.4-mini"] = {
    "input_cost_per_token": 0.75 / 1_000_000,
    "cache_read_input_token_cost": 0.075 / 1_000_000,
    "output_cost_per_token": 4.50 / 1_000_000,
    "litellm_provider": "openai",
}


TRAIN_PATH = Path(
    os.environ.get(
        "TRAIN_FILE",
        os.environ.get(
            "DEFAULT_TRAIN_FILE",
            "data/method_nav_prod_overnight_sample_50k.jsonl",
        ),
    )
)
TEST_PATH = Path(
    os.environ.get(
        "AUX_FILE",
        os.environ.get(
            "DEFAULT_AUX_FILE",
            "nav_1_3_24_gpt4o_relabeled - March 31, 2026 9_16_01 PM"
            " - c0fba847-9df3-4b8a-8190-5001cde7cc2e.jsonl",
        ),
    )
)

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
MODEL_NAME = os.environ.get("MODEL_NAME", "ivr-nav-ruler-pipeline")
PROJECT_NAME = os.environ.get("PROJECT", "genesisfi")

GRPO_GROUP_SIZE = int(os.environ.get("GRPO_GROUP_SIZE", "16"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "60"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-6"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
TRAIN_LIMIT = int(v) if (v := os.environ.get("TRAIN_LIMIT")) else None
MAX_STEPS = int(v) if (v := os.environ.get("MAX_STEPS")) else None
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
SAVE_CHECKPOINT = os.environ.get("SAVE_CHECKPOINT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIN_REWARD_STD = float(os.environ.get("MIN_REWARD_STD", "0.1"))
MAX_STEPS_OFF_POLICY = int(os.environ.get("MAX_STEPS_OFF_POLICY", "4"))
NUM_ROLLOUT_WORKERS = int(os.environ.get("NUM_ROLLOUT_WORKERS", "8"))
RULER_EVAL_CONCURRENCY = 50
GPT41_MODEL = "gpt-4.1"
RULER_JUDGE_MODEL = "openai/gpt-5.4"
PILOT_CONFIG = IVRNavPilotConfig.from_env(os.environ)
EVAL_LOG_DIR = Path(f"eval_logs/{MODEL_NAME}")


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_context_and_golden(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1]
    return messages, None


def insert_tool_responses(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        result.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            next_role = messages[idx + 1].get("role") if idx + 1 < len(messages) else None
            if next_role != "tool":
                for tc in msg["tool_calls"]:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "",
                        }
                    )
    return result


def choice_to_tool_name_and_args(choice: Choice) -> tuple[str, Any] | None:
    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        return None
    fn = tool_calls[0].function
    try:
        args = json.loads(fn.arguments) if fn.arguments else {}
    except Exception:
        args = fn.arguments
    return fn.name, args


def message_to_tool_name_and_args(msg: dict[str, Any]) -> tuple[str, Any] | None:
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        args = fn.get("arguments", {})
    return fn.get("name", ""), args


def tool_calls_match(choice: Choice, golden_msg: dict[str, Any]) -> bool:
    model_tc = choice_to_tool_name_and_args(choice)
    golden_tc = message_to_tool_name_and_args(golden_msg)
    if model_tc is None or golden_tc is None:
        return False
    return model_tc == golden_tc


def group_reward_std(group: art.TrajectoryGroup) -> float:
    rewards = [trajectory.reward for trajectory in group.trajectories]
    if len(rewards) < 2:
        return 0.0
    try:
        return statistics.stdev(rewards)
    except statistics.StatisticsError:
        return 0.0


def extract_final_turn(trajectory: art.Trajectory) -> dict[str, Any]:
    last = trajectory.messages_and_choices[-1] if trajectory.messages_and_choices else None
    if last is None:
        return {}
    if hasattr(last, "message"):
        msg = last.message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                )
        return {"content": msg.content, "tool_calls": tool_calls}
    return {
        "content": last.get("content", ""),
        "tool_calls": [
            {
                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "")),
            }
            for tc in (last.get("tool_calls") or [])
        ],
    }


def get_ruler_reasoning(trajectory: art.Trajectory) -> str | None:
    for entry in trajectory.logs or []:
        s = entry if isinstance(entry, str) else str(entry)
        if "RULER explanation:" in s:
            return s.split("RULER explanation:", 1)[1].strip()
    return None


def log_trajectory_samples(scored_groups: list[art.TrajectoryGroup], step: int) -> None:
    log_dir = EVAL_LOG_DIR / "trajectories"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"step_{step:04d}.jsonl"
    with open(log_path, "w") as f:
        for group in scored_groups:
            scenario_id = str(group.metadata.get("scenario_id", "")) if group.metadata else ""
            for idx, trajectory in enumerate(group.trajectories):
                entry = {
                    "step": step,
                    "scenario_id": scenario_id,
                    "trajectory_idx": idx,
                    "reward": trajectory.reward,
                    "ruler_reasoning": get_ruler_reasoning(trajectory),
                    "final_turn": extract_final_turn(trajectory),
                }
                f.write(json.dumps(entry) + "\n")


async def generate_response(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    temperature: float,
    n: int = 1,
) -> list[Choice]:
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                n=n,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices
        except Exception as exc:
            if "Already borrowed" in str(exc) and attempt < 4:
                await asyncio.sleep(2**attempt)
                continue
            print(f"    Warning: generation failed ({model_name}): {exc}")
            break
    return []


@track_api_cost(
    source="openai/gpt-4.1",
    provider="openai",
    model_name="openai/gpt-4.1",
)
async def generate_gpt41_response(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> ChatCompletion | None:
    try:
        return await client.chat.completions.create(
            model=GPT41_MODEL,
            messages=insert_tool_responses(messages),
            tools=tools or None,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        print(f"    Warning: GPT-4.1 generation failed: {exc}")
    return None


def build_runtime_config() -> dict[str, Any]:
    return {
        "pipeline_pilot": True,
        "project": PROJECT_NAME,
        "model_name": MODEL_NAME,
        "base_model": BASE_MODEL,
        "grpo_group_size": GRPO_GROUP_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "train_limit": TRAIN_LIMIT,
        "max_steps": MAX_STEPS,
        "shuffle_seed": PILOT_CONFIG.shuffle_seed,
        "n_holdout_rows": PILOT_CONFIG.n_holdout_rows,
        "n_test_rows": PILOT_CONFIG.n_test_rows,
        "max_tokens": MAX_TOKENS,
        "rollout_temperature": PILOT_CONFIG.rollout_temperature,
        "eval_temperature": PILOT_CONFIG.eval_temperature,
        "save_checkpoint": SAVE_CHECKPOINT,
        "ruler_judge_model": RULER_JUDGE_MODEL,
        "min_reward_std": MIN_REWARD_STD,
        "gpt41_model": GPT41_MODEL,
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "max_steps_off_policy": MAX_STEPS_OFF_POLICY,
        "num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "trainer_gpu_ids": _get_env_int_list("TRAINER_GPU_IDS", [0]),
        "inference_gpu_ids": _get_env_int_list("INFERENCE_GPU_IDS", [1]),
    }


async def run_eval(
    model: art.TrainableModel,
    model_client: AsyncOpenAI,
    gpt41_client: AsyncOpenAI,
    *,
    holdout_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    step: int,
    eval_temperature: float,
) -> None:
    print(f"\n{'─'*60}")
    print(f"Pilot eval at step {step}  (holdout={len(holdout_rows)}, test={len(test_rows)})")
    print(f"{'─'*60}")

    ruler_semaphore = asyncio.Semaphore(RULER_EVAL_CONCURRENCY)
    model_semaphore = asyncio.Semaphore(100)

    async def eval_holdout_row(row: dict[str, Any]) -> dict[str, Any] | None:
        messages = row["input"]["messages"]
        tools = row["input"].get("tools") or []

        async with model_semaphore:
            model_choices = await generate_response(
                model_client,
                model.get_inference_name(),
                messages,
                tools,
                temperature=eval_temperature,
            )
        model_choice = model_choices[0] if model_choices else None
        gpt41_resp = await generate_gpt41_response(gpt41_client, messages, tools)
        gpt41_choice = gpt41_resp.choices[0] if gpt41_resp and gpt41_resp.choices else None
        if model_choice is None or gpt41_choice is None:
            return None

        group = art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=list(messages) + [model_choice],
                    tools=tools,
                    reward=0.0,
                ),
                art.Trajectory(
                    messages_and_choices=list(messages) + [gpt41_choice],
                    tools=tools,
                    reward=0.0,
                ),
            ]
        )

        async with ruler_semaphore:
            scored = await ruler_score_group(
                group,
                RULER_JUDGE_MODEL,
                swallow_exceptions=True,
                debug=False,
            )
        if scored is None:
            return None

        model_score = scored.trajectories[0].reward
        gpt41_score = scored.trajectories[1].reward
        return {
            "win": model_score > gpt41_score,
            "tie": model_score == gpt41_score,
        }

    async def eval_test_row(row: dict[str, Any]) -> dict[str, Any] | None:
        messages = row["messages"]
        tools = row.get("tools") or []
        context, golden_msg = split_context_and_golden(messages)
        if golden_msg is None:
            return None

        async with model_semaphore:
            model_choices = await generate_response(
                model_client,
                model.get_inference_name(),
                context,
                tools,
                temperature=eval_temperature,
            )
        model_choice = model_choices[0] if model_choices else None
        if model_choice is None:
            return None

        return {"match": tool_calls_match(model_choice, golden_msg)}

    holdout_results, test_results = await asyncio.gather(
        tqdm.gather(
            *[eval_holdout_row(row) for row in holdout_rows],
            desc=f"Pilot eval (holdout) step {step}",
        ),
        tqdm.gather(
            *[eval_test_row(row) for row in test_rows],
            desc=f"Pilot eval (test) step {step}",
        ),
    )

    holdout_results = [row for row in holdout_results if row is not None]
    test_results = [row for row in test_results if row is not None]

    if holdout_results:
        wins = sum(1 for row in holdout_results if row["win"])
        ties = sum(1 for row in holdout_results if row["tie"])
        h_n = len(holdout_results)
        h_win_rate = wins / h_n
        h_tie_rate = ties / h_n
        h_win_tie_rate = (wins + ties) / h_n
    else:
        h_win_rate = h_tie_rate = h_win_tie_rate = float("nan")
        h_n = 0

    if test_results:
        t_match_rate = sum(1 for row in test_results if row["match"]) / len(test_results)
        t_n = len(test_results)
    else:
        t_match_rate = float("nan")
        t_n = 0

    print(f"  Holdout RULER win rate      : {h_win_rate:.1%}  (n={h_n})")
    print(f"  Holdout RULER tie rate      : {h_tie_rate:.1%}  (n={h_n})")
    print(f"  Holdout RULER win+tie rate  : {h_win_tie_rate:.1%}  (n={h_n})")
    print(f"  Test    match rate          : {t_match_rate:.1%}  (n={t_n})")

    await model.log(
        [],
        metrics={
            "ruler_win_rate": h_win_rate,
            "ruler_tie_rate": h_tie_rate,
            "ruler_win_tie_rate": h_win_tie_rate,
        },
        step=step,
        split="val",
    )
    await model.log(
        [],
        metrics={"test/golden_match_rate": t_match_rate},
        step=step,
        split="test",
    )

    EVAL_LOG_DIR.mkdir(exist_ok=True, parents=True)
    with open(EVAL_LOG_DIR / f"step_{step:04d}.json", "w") as f:
        json.dump(
            {
                "step": step,
                "holdout": {
                    "win_rate": h_win_rate,
                    "tie_rate": h_tie_rate,
                    "win_tie_rate": h_win_tie_rate,
                    "n": h_n,
                },
                "test": {"match_rate": t_match_rate, "n": t_n},
            },
            f,
        )


async def train(model: art.TrainableModel) -> None:
    runtime_config = build_runtime_config()
    print("\nRuntime config:")
    print(json.dumps(runtime_config, indent=2, sort_keys=True))
    model.update_wandb_config(runtime_config)

    backend = LocalBackend()
    gpt41_client = AsyncOpenAI(http_client=httpx.AsyncClient(timeout=90))

    await model.register(
        backend,
        _openai_client_config={
            "server_args": {
                "tool_call_parser": "qwen3_coder",
                "enable_auto_tool_choice": True,
            }
        },
    )
    model_client = model.openai_client()

    print(f"Loading training data from {TRAIN_PATH} ...")
    all_train = load_jsonl(TRAIN_PATH)
    print(f"  {len(all_train)} rows loaded")
    holdout_rows, train_rows = split_train_holdout_by_session(
        all_train,
        n_holdout_rows=PILOT_CONFIG.n_holdout_rows,
        shuffle_seed=PILOT_CONFIG.shuffle_seed,
    )
    if TRAIN_LIMIT is not None:
        train_rows = train_rows[:TRAIN_LIMIT]
    print(f"  {len(train_rows)} train / {len(holdout_rows)} holdout (session-aware)")

    print(f"Loading test data from {TEST_PATH} ...")
    all_test = load_jsonl(TEST_PATH)
    split_counts = Counter(str(row.get("split", "<missing>")) for row in all_test if "split" in row)
    if split_counts:
        print(f"  split distribution: {dict(split_counts)}")
    test_rows = select_test_rows(
        all_test,
        n_test_rows=PILOT_CONFIG.n_test_rows,
        shuffle_seed=PILOT_CONFIG.shuffle_seed + 1,
    )
    print(f"  {len(test_rows)} test rows selected")

    async def scenario_iter():
        for epoch in range(NUM_EPOCHS):
            epoch_rows = train_rows.copy()
            random.Random(PILOT_CONFIG.shuffle_seed + epoch).shuffle(epoch_rows)
            for idx, row in enumerate(epoch_rows):
                yield {
                    "row": row,
                    "metadata": {
                        "scenario_id": row_session_id(row, idx),
                        "epoch": epoch,
                    },
                }

    async def rollout_fn(
        rollout_model: art.TrainableModel,
        scenario: dict[str, Any],
        _config: IVRNavPilotConfig,
    ) -> art.TrajectoryGroup:
        row = scenario["row"]
        messages = row["input"]["messages"]
        tools = row["input"].get("tools") or []
        context, _ = split_context_and_golden(messages)

        choices = await generate_response(
            model_client,
            rollout_model.get_inference_name(),
            context,
            tools,
            temperature=PILOT_CONFIG.rollout_temperature,
            n=GRPO_GROUP_SIZE,
        )
        if not choices:
            group = art.TrajectoryGroup([])
            group.metadata["scenario_id"] = scenario["metadata"]["scenario_id"]
            return group

        group = art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=context + [choice],
                    tools=tools,
                    reward=0.0,
                )
                for choice in choices
            ]
        )
        group.metadata["scenario_id"] = scenario["metadata"]["scenario_id"]

        scored = await ruler_score_group(
            group,
            RULER_JUDGE_MODEL,
            swallow_exceptions=True,
            debug=False,
        )
        if scored is None or group_reward_std(scored) < MIN_REWARD_STD:
            empty_group = art.TrajectoryGroup([])
            empty_group.metadata = dict(group.metadata)
            return empty_group
        return scored

    async def eval_fn(
        eval_model: art.TrainableModel,
        step: int,
        _config: IVRNavPilotConfig,
    ) -> list[art.Trajectory]:
        await run_eval(
            eval_model,
            model_client,
            gpt41_client,
            holdout_rows=holdout_rows,
            test_rows=test_rows,
            step=step,
            eval_temperature=PILOT_CONFIG.eval_temperature,
        )
        return []

    trainer = PipelineTrainer(
        model=model,
        backend=backend,
        rollout_fn=rollout_fn,
        scenarios=scenario_iter(),
        config=PILOT_CONFIG,
        eval_fn=eval_fn,
        num_rollout_workers=NUM_ROLLOUT_WORKERS,
        min_batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        max_steps_off_policy=MAX_STEPS_OFF_POLICY,
        learning_rate=LEARNING_RATE,
        log_interval_seconds=float(os.environ.get("STATUS_LOG_INTERVAL_SECONDS", "60")),
        eval_every_n_steps=int(os.environ.get("EVAL_EVERY", "20")),
        eval_step_0=True,
        save_checkpoint=SAVE_CHECKPOINT,
        resume=SAVE_CHECKPOINT,
        max_steps=MAX_STEPS,
        total_scenarios=len(train_rows) * NUM_EPOCHS,
    )

    try:
        await trainer.train()
        step = await model.get_step()
        print(f"Pipeline pilot complete at step {step}")
    finally:
        await model_client.close()
        await gpt41_client.close()
        await backend.close()


if __name__ == "__main__":
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
        config=None,
        report_metrics=["wandb"],
    )
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "8192")),
            load_in_4bit=_get_env_bool("LOAD_IN_4BIT", False),
            load_in_16bit=_get_env_bool("LOAD_IN_16BIT", True),
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")),
            enforce_eager=_get_env_bool("ENFORCE_EAGER", True),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
            max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
        ),
        rollout_weights_mode=os.environ.get("ROLLOUT_WEIGHTS_MODE", "merged"),
        trainer_gpu_ids=_get_env_int_list("TRAINER_GPU_IDS", [0]),
        inference_gpu_ids=_get_env_int_list("INFERENCE_GPU_IDS", [1]),
    )
    asyncio.run(train(model))
