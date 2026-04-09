from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from typing import Any, Awaitable, Iterable, Literal, NamedTuple, TypeVar, cast
import os
import re
import time
import uuid

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.completion_usage import CompletionUsage
import tinker
import uvicorn

from art.tinker.cookbook_v import renderers, tokenizer_utils

from .. import dev
from ..backend import Backend
from ..costs import build_cost_calculator, compute_train_cost, get_model_pricing
from ..metrics_taxonomy import (
    build_training_summary_metrics,
    summarize_trajectory_groups,
)
from ..model import Model, TrainableModel
from ..tinker.backend import get_renderer_name
from ..tinker.server import get_free_port
from ..trajectories import TrajectoryGroup, get_messages
from ..types import TrainResult
from ..utils.output_dirs import get_model_dir
from ..utils.trajectory_migration import auto_migrate_on_register
from .data import (
    convert_openai_messages_to_renderer_format,
    parse_completion_to_openai_message,
    trajectory_groups_to_datums,
)

STATE_KEY_RUN_IDS = "tinker_run_ids"
STATE_KEY_LATEST_STEP = "latest_step"
T = TypeVar("T")

_UPSTREAM_TRAIN_METRIC_KEYS = {
    "reward": "reward",
    "reward_std_dev": "reward_std_dev",
    "exception_rate": "exception_rate",
    "policy_loss": "loss/train",
    "loss": "loss/train",
    "entropy": "loss/entropy",
    "kl_div": "loss/kl_div",
    "kl_policy_ref": "loss/kl_policy_ref",
    "grad_norm": "loss/grad_norm",
    "learning_rate": "loss/learning_rate",
    "num_groups_submitted": "data/step_num_groups_submitted",
    "num_groups_trainable": "data/step_num_groups_trainable",
    "num_trajectories": "data/step_num_trajectories",
    "num_trainable_tokens": "data/step_trainer_tokens",
    "train_tokens": "data/step_trainer_tokens",
    "num_datums": "data/step_num_datums",
}


def _canonicalize_upstream_metric_key(metric: str) -> str:
    if "/" in metric:
        return metric
    if metric == "tokens_per_second":
        return ""
    if metric.startswith("group_metric_"):
        return f"group_{metric[len('group_metric_') :]}"
    return _UPSTREAM_TRAIN_METRIC_KEYS.get(metric, metric)


class DistillationWorkItem(NamedTuple):
    """Work item for computing teacher logprobs in prompt distillation."""
    group_idx: int
    traj_idx: int
    prompt_tokens: list[int]
    completion_tokens: list[int]
    student_logprobs: list[float]
    student_prompt: Any
    teacher_prompt: Any
    prompt_messages: list[dict[str, Any]]
    teacher_system_prompt: str


@dataclass
class ModelState:
    service_client: tinker.ServiceClient
    rest_client: Any
    training_client: tinker.TrainingClient
    sampler_clients: dict[int, tinker.SamplingClient]
    sampler_checkpoint_paths: dict[int, str]
    training_checkpoint_paths: dict[int, str]
    current_step: int
    renderer: Any
    tokenizer: Any
    output_dir: str
    tinker_run_ids: list[str]
    model_name: str
    server_task: asyncio.Task[None] | None = None
    server_host: str | None = None
    server_port: int | None = None
    server_api_key: str | None = None


@dataclass
class TinkerNativeModelConfig:
    renderer_name: str
    training_client_args: dict[str, Any]


class TinkerNativeBackend(Backend):
    _tinker_train_log_env = "ART_TINKER_TRAIN_LOG"
    _tinker_sample_log_env = "ART_TINKER_SAMPLE_LOG"

    def __init__(
        self,
        *,
        tinker_api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        if not "TINKER_API_KEY" in os.environ or tinker_api_key is not None:
            assert tinker_api_key is not None, (
                "TINKER_API_KEY is not set and no tinker_api_key was provided"
            )
            print("Setting TINKER_API_KEY to", tinker_api_key, "in environment")
            os.environ["TINKER_API_KEY"] = tinker_api_key

        self._path = path or ".art"
        os.makedirs(self._path, exist_ok=True)
        self._model_state: dict[str, ModelState] = {}

    def _env_enabled(self, env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return False
        return value.lower() not in ("", "0", "false", "no")

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    async def _tinker_call(
        self,
        label: str,
        awaitable: Awaitable[T],
        *,
        env_name: str,
        prefix: str,
    ) -> T:
        if not self._env_enabled(env_name):
            return await awaitable
        start = time.perf_counter()
        print(f"[tinker:{prefix}] {label} start {self._timestamp()}")
        try:
            return await awaitable
        finally:
            elapsed = time.perf_counter() - start
            print(
                f"[tinker:{prefix}] {label} done in {elapsed:.2f}s "
                f"at {self._timestamp()}"
            )

    async def _tinker_train_call(self, label: str, awaitable: Awaitable[T]) -> T:
        return await self._tinker_call(
            label,
            awaitable,
            env_name=self._tinker_train_log_env,
            prefix="train",
        )

    async def _tinker_sample_call(self, label: str, awaitable: Awaitable[T]) -> T:
        return await self._tinker_call(
            label,
            awaitable,
            env_name=self._tinker_sample_log_env,
            prefix="sample",
        )

    async def close(self) -> None:
        for state in self._model_state.values():
            if state.server_task is not None:
                state.server_task.cancel()

    async def register(self, model: Model) -> None:
        model.base_path = self._path
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            import json

            json.dump(model.model_dump(), f)

        auto_migrate_on_register(output_dir)

        if not model.trainable:
            return
        trainable_model = cast(TrainableModel, model)
        pricing = get_model_pricing(trainable_model.base_model)
        if pricing is not None:
            trainable_model.set_cost_calculator(build_cost_calculator(pricing))
        state = await self._build_model_state(trainable_model)
        self._model_state[model.name] = state

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        state = self._model_state[model.name]

        raw_config: dict[str, Any] = cast(dict[str, Any], config) if config else {}
        server_args = cast(dict[str, Any], raw_config.get("server_args", {}))
        host = server_args.get("host", raw_config.get("host", "0.0.0.0"))
        port = server_args.get("port", raw_config.get("port"))
        if port is None:
            port = get_free_port()
        api_key = server_args.get("api_key", raw_config.get("api_key")) or "default"

        if state.server_task is None:
            state.server_host = host
            state.server_port = port
            state.server_api_key = api_key
            state.server_task = asyncio.create_task(
                self._run_openai_server(state, host=host, port=port)
            )
            state.server_task.add_done_callback(self._crash_on_server_exit)

        base_url = f"http://{host}:{port}/v1"
        await self._wait_for_server_ready(base_url, api_key, model)
        return base_url, api_key

    async def train(  # type: ignore[override]
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float = 1e-5,
        loss_fn: Literal["cispo", "ppo", "importance_sampling", "dro"] = "cispo",
        normalize_advantages: bool = True,
        save_checkpoint: bool = False,
        loss_fn_config: dict | None = None,
        adam_params: tinker.AdamParams | None = None,
    ) -> TrainResult:
        state = self._model_state[model.name]
        groups_list = list(trajectory_groups)
        summary = summarize_trajectory_groups(groups_list)

        datums = trajectory_groups_to_datums(
            groups_list,
            state.renderer,
            state.tokenizer,
            normalize_advantages,
        )

        metrics: dict[str, float] = {
            **build_training_summary_metrics(
                summary,
                include_trainable_groups=True,
            ),
            "data/step_num_datums": float(len(datums)),
        }

        if not datums:
            return TrainResult(step=state.current_step, metrics=metrics)

        train_tokens = 0
        for datum in datums:
            train_tokens += len(datum.model_input.to_ints())
        metrics["data/step_trainer_tokens"] = float(train_tokens)
        pricing = get_model_pricing(model.base_model)
        if pricing is not None:
            metrics["costs/train/tinker_train"] = compute_train_cost(
                train_tokens, pricing
            )
        trainer_started = time.monotonic()

        if adam_params is None:
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

        def remove_mask(datum: tinker.Datum) -> tinker.Datum:
            if "mask" not in datum.loss_fn_inputs:
                return datum
            loss_fn_inputs = {
                key: value
                for key, value in datum.loss_fn_inputs.items()
                if key != "mask"
            }
            return tinker.Datum(
                model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs
            )

        forward_output = await self._tinker_train_call(
            "forward_backward",
            state.training_client.forward_backward(
                [remove_mask(datum) for datum in datums],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        )
        optim_output = await self._tinker_train_call(
            "optim_step", state.training_client.optim_step(adam_params)
        )

        if forward_output.metrics:
            for key, value in forward_output.metrics.items():
                if value is None:
                    continue
                canonical_key = _canonicalize_upstream_metric_key(key)
                if canonical_key:
                    metrics[canonical_key] = float(value)
        if optim_output.metrics:
            for key, value in optim_output.metrics.items():
                if value is None:
                    continue
                canonical_key = _canonicalize_upstream_metric_key(key)
                if canonical_key:
                    metrics[canonical_key] = float(value)

        next_step = state.current_step + 1
        checkpoint_name = f"step_{next_step:06d}"

        if save_checkpoint:
            state_response, sampler_response = await self._save_checkpoint(
                state.training_client, checkpoint_name
            )
            state.training_checkpoint_paths[next_step] = state_response.path
        else:
            sampler_response = await self._save_sampler_weights(
                state.training_client, checkpoint_name
            )
        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            state.training_client.create_sampling_client_async(
                model_path=sampler_response.path
            ),
        )
        state.sampler_clients[next_step] = sampler_client
        state.sampler_checkpoint_paths[next_step] = sampler_response.path

        state.current_step = next_step
        self._persist_model_state(model, state)
        metrics["time/step_trainer_s"] = time.monotonic() - trainer_started

        return TrainResult(step=state.current_step, metrics=metrics)

    async def train_with_prompt_distillation(  # type: ignore[override]
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        teacher_system_prompts: str | list[str],
        learning_rate: float = 1e-5,
        loss_fn: str = "importance_sampling",
        normalize_advantages: bool = False,
        save_checkpoint: bool = False,
        loss_fn_config: dict | None = None,
        adam_params: tinker.AdamParams | None = None,
    ) -> TrainResult:
        """Train using on-policy distillation from the same model with modified system prompts.

        This implements distribution-level KL distillation where the teacher is the same model
        but with a modified system prompt. At each token position, per-token advantages are
        computed as log(p_student[token] / p_teacher[token]) and used with importance sampling loss.

        The teacher distributions are computed with stopgrad, so only the student model is updated.

        Note: Group size doesn't matter here since the loss is computed per-token based on
        distribution-level KL divergence. Groups of size 1 are typical.

        Args:
            model: The trainable model
            trajectory_groups: Trajectory groups sampled from the student
            teacher_system_prompts: Modified system prompt(s) to use for teacher distributions.
                - Single string: Same prompt used for all groups
                - List of strings: One prompt per group (must match number of groups)
            learning_rate: Learning rate for optimization
            loss_fn: Loss function to use (default: "importance_sampling")
            normalize_advantages: Ignored for distillation loss (kept for API compatibility)
            save_checkpoint: Whether to save full checkpoint
            loss_fn_config: Additional loss function configuration
            adam_params: Adam optimizer parameters
        """
        state = self._model_state[model.name]
        groups_list = list(trajectory_groups)

        # Convert single prompt to list
        if isinstance(teacher_system_prompts, str):
            prompts_list = [teacher_system_prompts] * len(groups_list)
        else:
            prompts_list = teacher_system_prompts
            if len(prompts_list) != len(groups_list):
                raise ValueError(
                    f"Number of teacher_system_prompts ({len(prompts_list)}) must match "
                    f"number of trajectory_groups ({len(groups_list)})"
                )

        # Compute teacher logprobs with modified system prompts and build datums
        datums = await self._trajectory_groups_to_datums_with_teacher_logprobs(
            groups_list,
            state,
            prompts_list,
            normalize_advantages,
        )

        metrics: dict[str, float] = {
            "num_groups_submitted": float(len(groups_list)),
            "num_datums": float(len(datums)),
        }

        if not datums:
            return TrainResult(step=state.current_step, metrics=metrics)

        train_tokens = 0
        for datum in datums:
            train_tokens += len(datum.model_input.to_ints())
        metrics["train_tokens"] = float(train_tokens)
        pricing = get_model_pricing(model.base_model)
        if pricing is not None:
            metrics["costs_train"] = compute_train_cost(train_tokens, pricing)

        if adam_params is None:
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

        # Use standard forward_backward with importance_sampling loss
        def remove_mask(datum: tinker.Datum) -> tinker.Datum:
            if "mask" not in datum.loss_fn_inputs:
                return datum
            loss_fn_inputs = {
                key: value
                for key, value in datum.loss_fn_inputs.items()
                if key != "mask"
            }
            return tinker.Datum(
                model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs
            )

        forward_output = await self._tinker_train_call(
            "forward_backward",
            state.training_client.forward_backward(
                [remove_mask(datum) for datum in datums],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        )

        optim_output = await self._tinker_train_call(
            "optim_step", state.training_client.optim_step(adam_params)
        )

        if forward_output.metrics:
            for key, value in forward_output.metrics.items():
                if value is None:
                    continue
                metrics[key] = float(value)
        if optim_output.metrics:
            for key, value in optim_output.metrics.items():
                if value is None:
                    continue
                metrics[key] = float(value)

        next_step = state.current_step + 1
        checkpoint_name = f"step_{next_step:06d}"

        if save_checkpoint:
            state_response, sampler_response = await self._save_checkpoint(
                state.training_client, checkpoint_name
            )
            state.training_checkpoint_paths[next_step] = state_response.path
        else:
            sampler_response = await self._save_sampler_weights(
                state.training_client, checkpoint_name
            )
        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            state.training_client.create_sampling_client_async(
                model_path=sampler_response.path
            ),
        )
        state.sampler_clients[next_step] = sampler_client
        state.sampler_checkpoint_paths[next_step] = sampler_response.path

        state.current_step = next_step
        self._persist_model_state(model, state)

        return TrainResult(step=state.current_step, metrics=metrics)

    async def _trajectory_groups_to_datums_with_teacher_logprobs(
        self,
        trajectory_groups: list[TrajectoryGroup],
        state: ModelState,
        teacher_system_prompts: list[str],
        normalize_advantages: bool,
    ) -> list[tinker.Datum]:
        """Convert trajectory groups to datums using distribution-level KL distillation.

        This computes teacher top-k distributions using the same model but with modified
        system prompts (one per group), then packages the full distributions for a
        distillation loss that computes KL(student || teacher) at each position.

        Note: Each trajectory is processed independently. Groups are only used to
        associate teacher prompts with trajectories.
        """
        from .data import (
            convert_openai_messages_to_renderer_format,
            extract_logprobs_from_choice,
            find_last_choice,
            iter_trajectory_histories,
        )
        import torch

        # Get the current sampler client (teacher)
        teacher_client = state.sampler_clients[state.current_step]

        # First pass: collect all work items
        work_items: list[DistillationWorkItem] = []

        for group_idx, (group, teacher_system_prompt) in enumerate(zip(trajectory_groups, teacher_system_prompts)):
            if not group.trajectories:
                continue

            # Process each trajectory in the group with this group's teacher prompt
            for traj_idx, trajectory in enumerate(group.trajectories):
                for history in iter_trajectory_histories(trajectory):
                    choice_info = find_last_choice(history.messages_and_choices)
                    if choice_info is None:
                        continue
                    choice_index, choice = choice_info

                    # Extract student logprobs and tokens from the choice
                    completion_tokens, student_logprobs = extract_logprobs_from_choice(
                        choice, state.tokenizer
                    )
                    if not completion_tokens or len(completion_tokens) != len(student_logprobs):
                        continue

                    # Build student prompt (original)
                    prompt_messages = cast(
                        list[dict[str, Any]],
                        get_messages(history.messages_and_choices[:choice_index]),
                    )
                    student_renderer_messages = convert_openai_messages_to_renderer_format(
                        messages=prompt_messages,
                        tools=cast(list[dict[str, Any]] | None, history.tools),
                        renderer=state.renderer,
                    )
                    student_prompt = state.renderer.build_generation_prompt(
                        student_renderer_messages
                    )
                    prompt_tokens = list(student_prompt.to_ints())

                    # Build teacher prompt with modified system prompt
                    teacher_messages = self._modify_system_prompt(
                        prompt_messages, teacher_system_prompt
                    )
                    teacher_renderer_messages = convert_openai_messages_to_renderer_format(
                        messages=teacher_messages,
                        tools=cast(list[dict[str, Any]] | None, history.tools),
                        renderer=state.renderer,
                    )
                    teacher_prompt = state.renderer.build_generation_prompt(
                        teacher_renderer_messages
                    )

                    work_items.append(DistillationWorkItem(
                        group_idx=group_idx,
                        traj_idx=traj_idx,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        student_logprobs=student_logprobs,
                        student_prompt=student_prompt,
                        teacher_prompt=teacher_prompt,
                        prompt_messages=prompt_messages,
                        teacher_system_prompt=teacher_system_prompt,
                    ))

        # Second pass: compute all top-k distributions in batch
        if not work_items:
            return []

        print(f"Computing top-k distributions for {len(work_items)} trajectories in batch...")
        distribution_results = await self._compute_topk_distributions_batch(
            teacher_client,
            [(item.student_prompt, item.teacher_prompt, item.completion_tokens) for item in work_items],
        )

        # Third pass: build datums with full distributions for distillation loss
        datums: list[tinker.Datum] = []
        all_kl_values: list[float] = []
        trajectory_count = 0

        for item, dist_result in zip(work_items, distribution_results):
            if dist_result is None:
                continue

            student_topk, teacher_topk, student_tail_mass, teacher_tail_mass = dist_result

            # Compute KL for logging only
            kl_per_token = []
            for s_topk, t_topk, s_tail, t_tail in zip(student_topk, teacher_topk, student_tail_mass, teacher_tail_mass):
                kl = self._compute_topk_kl_single_with_tail(s_topk, t_topk, s_tail, t_tail)
                kl_per_token.append(kl)
            all_kl_values.extend(kl_per_token)

            # Log first trajectory for debugging
            if item.group_idx == 0 and item.traj_idx == 0 and len(datums) == 0:
                print("\n[Prompt Distillation Logging]")
                print("=" * 80)

                # Decode and show completion text
                completion_text = state.tokenizer.decode(item.completion_tokens)
                print(f"Completion text: {repr(completion_text)}")
                print(f"Completion tokens ({len(item.completion_tokens)}): {item.completion_tokens[:10]}...")

                # Show student prompt (original)
                student_msg_content = item.prompt_messages[0].get("content", "")
                student_system = next(
                    (m.get("content", "") for m in item.prompt_messages if m.get("role") == "system"),
                    "(no system prompt)"
                )
                print(f"\nStudent system: {student_system}")
                print(f"Student user msg: {student_msg_content}...")

                # Show teacher prompt (modified)
                print(f"\nTeacher system: {item.teacher_system_prompt}")

                # Debug: Show prompt structure
                print(f"\n[Trajectory Context Debug]")
                print(f"Number of messages in prompt: {len(item.prompt_messages)}")
                for idx, msg in enumerate(item.prompt_messages):
                    role = msg.get('role', '?')
                    content_preview = str(msg.get('content', ''))[:100]
                    print(f"  Message {idx} ({role}): {content_preview}...")
                print(f"Completion tokens length: {len(item.completion_tokens)}")
                print(f"Completion text: {repr(state.tokenizer.decode(item.completion_tokens)[:200])}...")

                # Show sample distributions and KL
                print("\nSample distributions:")
                for i in range(len(student_topk)):
                    actual_id = item.completion_tokens[i]
                    actual_txt = state.tokenizer.decode([actual_id])

                    s_top3 = ", ".join(
                        f"{state.tokenizer.decode([tid])!r}(id={tid}):lp={lp:.4f},p={math.exp(lp):.4f}"
                        for tid, lp in student_topk[i][:3]
                    )
                    t_top3 = ", ".join(
                        f"{state.tokenizer.decode([tid])!r}(id={tid}):lp={lp:.4f},p={math.exp(lp):.4f}"
                        for tid, lp in teacher_topk[i][:3]
                    )

                    print(
                        f"pos={i} actual={actual_txt!r}(id={actual_id}) | "
                        f"S[{s_top3}] tail={student_tail_mass[i]:.4f} | "
                        f"T[{t_top3}] tail={teacher_tail_mass[i]:.4f} | "
                        f"KL={kl_per_token[i]:.4f}"
                    )

            # Build k datums (one per rank) for mean-seeking KL distillation
            traj_datums = self._build_datums_with_distributions(
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                student_topk=student_topk,
                teacher_topk=teacher_topk,
                student_tail_mass=student_tail_mass,
                teacher_tail_mass=teacher_tail_mass,
                trajectory_id=trajectory_count,
            )
            datums.extend(traj_datums)
            trajectory_count += 1

        # Summary statistics
        if all_kl_values:
            import numpy as np
            kl_array = np.array(all_kl_values)

            print(f"\n[Distillation Statistics]")
            print(f"Total tokens: {len(all_kl_values)}")
            print(f"Top-k KL(student||teacher) - mean: {kl_array.mean():.4f}, std: {kl_array.std():.4f}, min: {kl_array.min():.4f}, max: {kl_array.max():.4f}")
            print("=" * 80)

        return datums

    def _modify_system_prompt(
        self, messages: list[dict[str, Any]], new_system_prompt: str
    ) -> list[dict[str, Any]]:
        """Ensure the system prompt is inserted right before the last message."""
        # Remove any existing system messages
        modified = [m for m in messages if m.get("role") != "system"]

        system_msg = {"role": "system", "content": new_system_prompt}

        # Insert right before the last message (or at start if there's no "last")
        insert_at = max(len(modified) - 1, 0)
        modified.insert(insert_at, system_msg)

        return modified


    async def _compute_topk_distributions_batch(
        self,
        sampler_client: tinker.SamplingClient,
        items: list[tuple[Any, Any, list[int]]],  # [(student_prompt, teacher_prompt, completion_tokens), ...]
        k: int = 20,
    ) -> list[tuple[list[list[tuple[int, float]]], list[list[tuple[int, float]]], list[float], list[float]] | None]:
        """Compute top-k distributions for student and teacher for multiple items in parallel.

        Returns:
            List of (student_topk, teacher_topk, student_tail_mass, teacher_tail_mass) or None
            where student_topk/teacher_topk are lists (per position) of lists of (token_id, logprob) tuples,
            and student_tail_mass/teacher_tail_mass are lists (per position) of tail probabilities.
        """
        import asyncio
        import math

        # Kick off all sample calls (returns futures immediately)
        futures_student = []
        futures_teacher = []

        # Store prompt lengths for proper indexing
        student_prompt_lens = []
        teacher_prompt_lens = []

        for student_prompt, teacher_prompt, completion_tokens in items:
            student_prompt_len = len(student_prompt.to_ints())
            teacher_prompt_len = len(teacher_prompt.to_ints())

            student_prompt_lens.append(student_prompt_len)
            teacher_prompt_lens.append(teacher_prompt_len)

            student_tokens = list(student_prompt.to_ints()) + completion_tokens
            teacher_tokens = list(teacher_prompt.to_ints()) + completion_tokens

            # Note: max_tokens=1 (minimum allowed), we'll ignore the generated token
            params = tinker.SamplingParams(max_tokens=1, temperature=0.0)

            futures_student.append(
                sampler_client.sample(
                    prompt=tinker.ModelInput.from_ints(student_tokens),
                    num_samples=1,
                    sampling_params=params,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=k,
                )
            )
            futures_teacher.append(
                sampler_client.sample(
                    prompt=tinker.ModelInput.from_ints(teacher_tokens),
                    num_samples=1,
                    sampling_params=params,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=k,
                )
            )

        # Await all results in parallel
        async def get_distributions_result(student_future, teacher_future, student_prompt_len, teacher_prompt_len, completion_len, completion_tokens, tokenizer):
            try:
                student_resp = await asyncio.to_thread(student_future.result)
                teacher_resp = await asyncio.to_thread(teacher_future.result)

                # Extract the correct positions
                student_start = student_prompt_len
                student_end = student_start + completion_len
                teacher_start = teacher_prompt_len
                teacher_end = teacher_start + completion_len

                student_topk = student_resp.topk_prompt_logprobs[student_start:student_end]
                teacher_topk = teacher_resp.topk_prompt_logprobs[teacher_start:teacher_end]

                # Compute tail masses for each position
                import math
                student_tail_mass = []
                teacher_tail_mass = []

                for s_topk in student_topk:
                    # Sum probabilities in top-k
                    topk_prob_sum = sum(math.exp(lp) for _, lp in s_topk)
                    # Tail mass is the remaining probability
                    student_tail_mass.append(max(1e-10, 1.0 - topk_prob_sum))

                for t_topk in teacher_topk:
                    topk_prob_sum = sum(math.exp(lp) for _, lp in t_topk)
                    teacher_tail_mass.append(max(1e-10, 1.0 - topk_prob_sum))

                return (student_topk, teacher_topk, student_tail_mass, teacher_tail_mass)
            except Exception as e:
                print(f"Error computing top-k distributions: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Get tokenizer from sampler_client
        tokenizer = sampler_client.get_tokenizer()

        results = await asyncio.gather(*[
            get_distributions_result(
                futures_student[i],
                futures_teacher[i],
                student_prompt_lens[i],
                teacher_prompt_lens[i],
                len(items[i][2]),
                items[i][2],  # completion_tokens
                tokenizer
            )
            for i in range(len(items))
        ])

        return results

    def _compute_topk_kl_single_with_tail(
        self,
        student_topk: list[tuple[int, float]],
        teacher_topk: list[tuple[int, float]],
        student_tail_mass: float,
        teacher_tail_mass: float,
    ) -> float:
        """Compute KL divergence using top-k approximation with explicit tail masses.

        KL(P||Q) ≈ Σ p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)

        For tokens in student top-k but not in teacher top-k, we distribute the
        teacher tail mass proportionally among them.
        """
        import math

        # Convert to dicts for easy lookup
        student_dict = {tok: logp for tok, logp in student_topk}
        teacher_dict = {tok: logp for tok, logp in teacher_topk}

        # Convert logprobs to probs
        student_probs = {tok: math.exp(lp) for tok, lp in student_dict.items()}
        teacher_probs = {tok: math.exp(lp) for tok, lp in teacher_dict.items()}

        # Compute KL for top-k tokens
        kl = 0.0
        for tok, p_s in student_probs.items():
            if p_s < 1e-10:
                continue
            # If token not in teacher top-k, assign it a share of the tail
            p_t = teacher_probs.get(tok, teacher_tail_mass / max(1, len(student_probs)))
            kl += p_s * math.log(p_s / max(1e-10, p_t))

        # Add tail contribution
        if student_tail_mass > 1e-10:
            kl += student_tail_mass * math.log(student_tail_mass / max(1e-10, teacher_tail_mass))

        return kl

    async def _compute_logprobs_for_completion(
        self,
        sampler_client: tinker.SamplingClient,
        prompt: Any,
        completion_tokens: list[int],
    ) -> list[float] | None:
        """Compute logprobs for specific completion tokens given a prompt.

        This uses the sampler's compute_logprobs method to compute logprobs
        for the completion tokens conditioned on the prompt.
        """
        # Concatenate prompt and completion tokens
        prompt_tokens = list(prompt.to_ints())
        all_tokens = prompt_tokens + completion_tokens

        # Create ModelInput with all tokens
        model_input = tinker.ModelInput.from_ints(tokens=all_tokens)

        # Compute logprobs for all tokens
        # all_logprobs[i] is the logprob of token[i] given tokens[0:i]
        future = sampler_client.compute_logprobs(model_input)
        all_logprobs = future.result()

        # Extract logprobs for completion tokens only
        # Completion tokens start at index len(prompt_tokens)
        # e.g., if prompt = [p1, p2, p3] and completion = [c1, c2]
        # all_tokens = [p1, p2, p3, c1, c2]
        # all_logprobs[3] = logprob of c1 given [p1, p2, p3]
        # all_logprobs[4] = logprob of c2 given [p1, p2, p3, c1]
        completion_start_idx = len(prompt_tokens)
        completion_logprobs = all_logprobs[completion_start_idx:completion_start_idx + len(completion_tokens)]

        # Check for None values (failed to compute)
        if any(lp is None for lp in completion_logprobs):
            return None

        return cast(list[float], completion_logprobs)

    def _build_datums_with_distributions(
        self,
        prompt_tokens: list[int],
        completion_tokens: list[int],
        student_topk: list[list[tuple[int, float]]],
        teacher_topk: list[list[tuple[int, float]]],
        student_tail_mass: list[float],
        teacher_tail_mass: list[float],
        trajectory_id: int = 0,
    ) -> list[tinker.Datum]:
        """Build k datums for top-k SDPO with probability-weighted advantages.

        Creates k datums where datum[r] contains:
        - target_tokens: [seq_len] - rank-r token from STUDENT's top-k at each position
        - advantages: [seq_len] - p_student[r] * (log p_teacher[r] - log p_student[r])
        - logprobs: [seq_len] - student's logprob (for importance sampling)

        The probability weighting ensures that when losses are summed across all k datums,
        we approximate: -KL(student || teacher) = -Σ p_student * log(p_student / p_teacher)
                                                 = Σ p_student * (log p_teacher - log p_student)

        Iterates through student's top-k (mean-seeking KL) and finds corresponding teacher probs.
        Uses standard importance_sampling loss with per-token advantages.
        """
        import torch
        import math

        if not prompt_tokens or not completion_tokens:
            return []

        k = min(5, len(student_topk[0]) if student_topk else 20)  # Limit k to avoid too many datums

        # Build input tokens
        ob_len = max(len(prompt_tokens) - 1, 0)
        all_tokens = prompt_tokens + completion_tokens
        input_tokens = all_tokens[:-1]
        completion_len = len(completion_tokens)
        seq_len = len(input_tokens)

        if seq_len != ob_len + completion_len:
            return []

        # Create k datums, one for each rank in student's top-k
        datums = []
        for rank in range(k):
            target_tokens_list = []
            student_logprobs_list = []
            advantages_list = []
            mask_list = []
            has_nonzero_prob = False

            for pos_idx in range(seq_len):
                if pos_idx < ob_len:
                    # Prompt position: dummy values
                    target_tokens_list.append(0)
                    student_logprobs_list.append(0.0)
                    advantages_list.append(0.0)
                    mask_list.append(0.0)
                else:
                    # Completion position: rank-r token from STUDENT's top-k
                    completion_pos = pos_idx - ob_len
                    student_topk_at_pos = student_topk[completion_pos]
                    teacher_topk_at_pos = teacher_topk[completion_pos]

                    if rank < len(student_topk_at_pos):
                        # Student's rank-r token
                        student_tok_id, student_logp = student_topk_at_pos[rank]
                        student_prob = math.exp(student_logp)

                        # Check if this rank has near-zero probability (numerical issues)
                        if student_prob < 1e-10:
                            # This rank and higher have negligible probability, skip
                            target_tokens_list.append(0)
                            student_logprobs_list.append(0.0)
                            advantages_list.append(0.0)
                            mask_list.append(0.0)
                            continue

                        has_nonzero_prob = True

                        # Find this token's logprob in teacher's top-k
                        teacher_logp = None
                        for t_tok_id, t_logp in teacher_topk_at_pos:
                            if t_tok_id == student_tok_id:
                                teacher_logp = t_logp
                                break

                        if teacher_logp is None:
                            # Token not in teacher's top-k, estimate from tail
                            # Distribute tail mass uniformly among missing tokens
                            teacher_logp = math.log(max(1e-10, teacher_tail_mass[completion_pos] / max(1, k)))

                        # PROBABILITY-WEIGHTED advantage for top-k KL approximation
                        # advantage = p_student * (log p_teacher - log p_student)
                        # This ensures the sum over k datums approximates KL divergence
                        # The probability weighting naturally limits the magnitude, so no clamping needed
                        advantage = student_prob * (teacher_logp - student_logp)

                        target_tokens_list.append(int(student_tok_id))
                        student_logprobs_list.append(float(student_logp))
                        advantages_list.append(advantage)
                        mask_list.append(1.0)
                    else:
                        # Less than k tokens at this position
                        target_tokens_list.append(0)
                        student_logprobs_list.append(0.0)
                        advantages_list.append(0.0)
                        mask_list.append(0.0)

            # Skip this datum if it has no non-zero probabilities (all ranks exhausted)
            if not has_nonzero_prob:
                break

            # Debug logging for first datum
            if rank == 0 and trajectory_id == 0:
                print(f"\n[SDPO Datum Creation Debug]")
                print(f"seq_len: {seq_len}, k: {k}, ob_len: {ob_len}, completion_len: {completion_len}")
                # Only show positions that exist
                n_show = min(3, completion_len)
                if n_show > 0:
                    print(f"Sample advantages (first {n_show} completion positions): {advantages_list[ob_len:ob_len+n_show]}")
                    print(f"Sample student probs (first {n_show} completion positions): {[math.exp(student_logprobs_list[i]) for i in range(ob_len, ob_len+n_show)]}")

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(
                        torch.tensor(target_tokens_list, dtype=torch.int64)
                    ),
                    "logprobs": tinker.TensorData.from_torch(
                        torch.tensor(student_logprobs_list, dtype=torch.float32)
                    ),
                    "advantages": tinker.TensorData.from_torch(
                        torch.tensor(advantages_list, dtype=torch.float32)
                    ),
                    "mask": tinker.TensorData.from_torch(
                        torch.tensor(mask_list, dtype=torch.float32)
                    ),
                },
            )

            datums.append(datum)

        return datums

    def _build_datum_with_advantages(
        self,
        prompt_tokens: list[int],
        completion_tokens: list[int],
        student_logprobs: list[float],
        advantages: list[float],
    ) -> tinker.Datum | None:
        """Build a tinker Datum with custom per-token advantages."""
        import torch

        if not prompt_tokens or not completion_tokens:
            return None

        ob_len = max(len(prompt_tokens) - 1, 0)
        all_tokens = prompt_tokens + completion_tokens
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]

        padded_logprobs = [0.0] * ob_len + list(student_logprobs)
        padded_advantages = [0.0] * ob_len + list(advantages)
        action_mask = [0.0] * ob_len + [1.0] * len(completion_tokens)

        if not (
            len(input_tokens)
            == len(target_tokens)
            == len(padded_logprobs)
            == len(padded_advantages)
            == len(action_mask)
        ):
            return None

        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(padded_logprobs, dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.tensor(padded_advantages, dtype=torch.float32)
                ),
                "mask": tinker.TensorData.from_torch(
                    torch.tensor(action_mask, dtype=torch.float32)
                ),
            },
        )

    async def _get_step(self, model: TrainableModel) -> int:
        if model.name in self._model_state:
            return self._model_state[model.name].current_step
        state = model.read_state() or {}
        return int(state.get(STATE_KEY_LATEST_STEP, 0))

    async def _delete_checkpoint_files(
        self,
        model: TrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        print("Checkpoint deletion is not yet implemented for TinkerNativeBackend.")

    def _model_inference_name(self, model: Model, step: int | None = None) -> str:
        base_name = model.inference_model_name or model.name
        if "@" in base_name:
            base_name = base_name.split("@", 1)[0]
        if step is None:
            state = self._model_state.get(model.name)
            step = state.current_step if state is not None else 0
        return f"{base_name}@{step}"

    async def _run_openai_server(
        self,
        state: ModelState,
        host: str,
        port: int,
    ) -> None:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(body: CompletionCreateParams) -> ChatCompletion:
            model_name = body.get("model")
            parsed_model_name, step = self._parse_model_name(model_name)
            sampler_client = await self._get_sampler_client(state, step)

            messages = self._normalize_messages(body["messages"])
            tools = self._normalize_tools(body.get("tools"))
            renderer_messages = convert_openai_messages_to_renderer_format(
                messages=messages,
                tools=tools,
                renderer=state.renderer,
            )
            prompt = state.renderer.build_generation_prompt(renderer_messages)
            prompt_tokens = list(prompt.to_ints())

            max_tokens = body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = body.get("max_tokens")
            temperature = body.get("temperature")
            top_k = body.get("top_k")
            top_p = body.get("top_p")
            sampling_params = tinker.SamplingParams(
                max_tokens=max_tokens,
                seed=body.get("seed"),
                temperature=temperature if temperature is not None else 1.0,
                top_k=top_k if top_k is not None else -1,
                top_p=top_p if top_p is not None else 1.0,
                stop=state.renderer.get_stop_sequences(),
            )
            sample_response = await self._tinker_sample_call(
                "sample_async",
                sampler_client.sample_async(
                    prompt=prompt,
                    num_samples=body.get("n") or 1,
                    sampling_params=sampling_params,
                ),
            )

            choices: list[Choice] = []
            for i, sequence in enumerate(sample_response.sequences):
                if sequence.logprobs is None:
                    raise HTTPException(status_code=400, detail="Logprobs are required")
                if len(sequence.tokens) != len(sequence.logprobs):
                    raise HTTPException(
                        status_code=400,
                        detail="Tokens and logprobs must have the same length",
                    )
                parsed_message = parse_completion_to_openai_message(
                    list(sequence.tokens), state.renderer
                )
                content = parsed_message.get("content")
                tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None
                if parsed_message.get("tool_calls"):
                    tool_calls = [
                        ChatCompletionMessageFunctionToolCall(
                            type="function",
                            id=tool_call.get("id") or f"call_{idx}",
                            function=Function(
                                name=tool_call["function"]["name"],
                                arguments=(
                                    tool_call["function"]["arguments"]
                                    if isinstance(
                                        tool_call["function"]["arguments"], str
                                    )
                                    else json.dumps(tool_call["function"]["arguments"])
                                ),
                            ),
                        )
                        for idx, tool_call in enumerate(parsed_message["tool_calls"])
                    ]
                choices.append(
                    Choice(
                        finish_reason=sequence.stop_reason,
                        index=i,
                        message=ChatCompletionMessage(
                            content=content or None,
                            role="assistant",
                            tool_calls=tool_calls,
                        ),
                        logprobs=ChoiceLogprobs(
                            content=[
                                ChatCompletionTokenLogprob(
                                    token=f"token_id:{token}",
                                    logprob=logprob,
                                    top_logprobs=[],
                                )
                                for token, logprob in zip(
                                    sequence.tokens, sequence.logprobs
                                )
                            ]
                        ),
                    )
                )

            completion_tokens = sum(
                len(sequence.tokens) for sequence in sample_response.sequences
            )
            return ChatCompletion(
                id=str(uuid.uuid4()),
                choices=choices,
                created=int(time.time()),
                model=self._format_response_model(parsed_model_name, step),
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=len(prompt_tokens),
                    total_tokens=completion_tokens + len(prompt_tokens),
                ),
            )

        server_config = uvicorn.Config(app, host=host, port=port, log_level="error")
        server = uvicorn.Server(server_config)
        await server.serve()

    def _crash_on_server_exit(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"OpenAI server crashed: {exc}")
        else:
            print("OpenAI server exited unexpectedly.")
        os._exit(1)

    async def _wait_for_server_ready(
        self, base_url: str, api_key: str, model: TrainableModel
    ) -> None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        with_timeout = float(os.environ.get("ART_SERVER_TIMEOUT", 300.0))
        start = time.time()
        while True:
            if time.time() - start > with_timeout:
                raise TimeoutError(
                    f"Unable to reach OpenAI-compatible server within {with_timeout} seconds."
                )
            try:
                await client.chat.completions.create(
                    model=self._model_inference_name(model),
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    max_completion_tokens=1,
                )
                return
            except Exception:
                await asyncio.sleep(0.1)

    async def _build_model_state(self, model: TrainableModel) -> ModelState:
        config = self._resolve_model_config(model)
        service_client = tinker.ServiceClient()
        rest_client = service_client.create_rest_client()

        tokenizer = tokenizer_utils.get_tokenizer(model.base_model)
        renderer = renderers.get_renderer(
            name=config.renderer_name,
            tokenizer=tokenizer,
            model_name=model.base_model,
        )

        saved_state = model.read_state() or {}
        tinker_run_ids = list(saved_state.get(STATE_KEY_RUN_IDS, []))
        training_paths, sampler_paths = await self._list_checkpoints(
            rest_client, tinker_run_ids
        )

        training_client: tinker.TrainingClient
        current_step = 0

        if training_paths:
            current_step = max(training_paths.keys())
            checkpoint_path = training_paths[current_step]
            training_client = await self._create_training_client_from_checkpoint(
                service_client=service_client,
                checkpoint_state_path=checkpoint_path,
                base_model=model.base_model,
                training_client_args=config.training_client_args,
                reset_optimizer=False,
            )
        else:
            training_client = await self._tinker_train_call(
                "create_lora_training_client_async",
                service_client.create_lora_training_client_async(
                    model.base_model, **config.training_client_args
                ),
            )
            checkpoint_name = f"step_{current_step:06d}"
            sampler_response = await self._save_sampler_weights(
                training_client, checkpoint_name
            )
            sampler_paths[current_step] = sampler_response.path

        run_id = training_client.model_id
        if run_id not in tinker_run_ids:
            tinker_run_ids.append(run_id)

        sampler_clients: dict[int, tinker.SamplingClient] = {}
        if current_step in sampler_paths:
            sampler_clients[current_step] = await self._tinker_train_call(
                "create_sampling_client_async",
                training_client.create_sampling_client_async(
                    model_path=sampler_paths[current_step]
                ),
            )
        else:
            checkpoint_name = f"step_{current_step:06d}"
            sampler_response = await self._save_sampler_weights(
                training_client, checkpoint_name
            )
            sampler_paths[current_step] = sampler_response.path
            sampler_clients[current_step] = await self._tinker_train_call(
                "create_sampling_client_async",
                training_client.create_sampling_client_async(
                    model_path=sampler_response.path
                ),
            )

        state = ModelState(
            service_client=service_client,
            rest_client=rest_client,
            training_client=training_client,
            sampler_clients=sampler_clients,
            sampler_checkpoint_paths=sampler_paths,
            training_checkpoint_paths=training_paths,
            current_step=current_step,
            renderer=renderer,
            tokenizer=tokenizer,
            output_dir=get_model_dir(model=model, art_path=self._path),
            tinker_run_ids=tinker_run_ids,
            model_name=((model.inference_model_name or model.name).split("@", 1)[0]),
        )
        self._persist_model_state(model, state)
        return state

    def _resolve_model_config(self, model: TrainableModel) -> TinkerNativeModelConfig:
        internal_config = model._internal_config or {}
        tinker_native_args = cast(
            dev.TinkerNativeArgs | None,
            internal_config.get("tinker_native_args"),
        )
        renderer_name = (
            tinker_native_args.get("renderer_name")
            if tinker_native_args is not None
            else None
        )
        if renderer_name is None:
            renderer_name = get_renderer_name(model.base_model)

        training_client_args = dict(
            tinker_native_args.get("training_client_args", {})
            if tinker_native_args is not None
            else {}
        )
        if "rank" not in training_client_args:
            training_client_args["rank"] = 8
        if "train_unembed" not in training_client_args:
            training_client_args["train_unembed"] = False

        return TinkerNativeModelConfig(
            renderer_name=renderer_name,
            training_client_args=training_client_args,
        )

    async def _list_checkpoints(
        self, rest_client: Any, tinker_run_ids: list[str]
    ) -> tuple[dict[int, str], dict[int, str]]:
        training_paths: dict[int, str] = {}
        sampler_paths: dict[int, str] = {}
        step_pattern = re.compile(r"(?:weights/)?step_(\d+)$")

        for run_id in tinker_run_ids:
            try:
                response = await self._tinker_train_call(
                    f"list_checkpoints_async {run_id}",
                    rest_client.list_checkpoints_async(run_id),
                )
            except Exception as exc:
                print(f"Warning: Could not list checkpoints for {run_id}: {exc}")
                continue
            for checkpoint in response.checkpoints:
                match = step_pattern.match(checkpoint.checkpoint_id)
                if not match:
                    continue
                step = int(match.group(1))
                path = f"tinker://{run_id}/{checkpoint.checkpoint_id}"
                if checkpoint.checkpoint_type == "training":
                    training_paths[step] = path
                elif checkpoint.checkpoint_type == "sampler":
                    sampler_paths[step] = path
        return training_paths, sampler_paths

    async def _get_sampler_client(
        self, state: ModelState, step: int | None
    ) -> tinker.SamplingClient:
        actual_step = step if step is not None else state.current_step
        if actual_step in state.sampler_clients:
            return state.sampler_clients[actual_step]

        if actual_step not in state.sampler_checkpoint_paths:
            training_paths, sampler_paths = await self._list_checkpoints(
                state.rest_client, state.tinker_run_ids
            )
            state.training_checkpoint_paths.update(training_paths)
            state.sampler_checkpoint_paths.update(sampler_paths)

        if actual_step not in state.sampler_checkpoint_paths:
            available = sorted(state.sampler_checkpoint_paths.keys())
            raise HTTPException(
                status_code=404,
                detail=f"No sampler checkpoint for step {actual_step}. Available: {available}",
            )

        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            state.training_client.create_sampling_client_async(
                model_path=state.sampler_checkpoint_paths[actual_step]
            ),
        )
        state.sampler_clients[actual_step] = sampler_client
        return sampler_client

    def _normalize_messages(self, messages: Iterable[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if hasattr(message, "model_dump"):
                normalized.append(message.model_dump())
            else:
                normalized.append(dict(message))
        return normalized

    def _normalize_tools(
        self, tools: Iterable[Any] | None
    ) -> list[dict[str, Any]] | None:
        if tools is None:
            return None
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if hasattr(tool, "model_dump"):
                normalized.append(tool.model_dump())
            else:
                normalized.append(dict(tool))
        return normalized

    def _parse_model_name(self, model_name: str | None) -> tuple[str, int]:
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Model name is required and must include an '@step' suffix. Use model.get_inference_name().",
            )
        if "@" not in model_name:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model_name}' is missing an '@step' suffix. "
                    "Use model.get_inference_name()."
                ),
            )

        base_name, step_str = model_name.rsplit("@", 1)
        try:
            return base_name, int(step_str)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid model step: {model_name}"
            ) from exc

    def _format_response_model(self, model_name: str, step: int) -> str:
        # Echo back the explicit model@step used for this completion.
        return f"{model_name}@{step}"

    async def _create_training_client_from_checkpoint(
        self,
        service_client: tinker.ServiceClient,
        checkpoint_state_path: str,
        base_model: str,
        training_client_args: dict[str, Any],
        reset_optimizer: bool = False,
    ) -> tinker.TrainingClient:
        training_client = await self._tinker_train_call(
            "create_lora_training_client_async",
            service_client.create_lora_training_client_async(
                base_model, **training_client_args
            ),
        )

        if reset_optimizer:
            load_future = await self._tinker_train_call(
                "load_state_async",
                training_client.load_state_async(checkpoint_state_path),
            )
            await self._tinker_train_call(
                "load_state_result_async", load_future.result_async()
            )
        else:
            load_future = await self._tinker_train_call(
                "load_state_with_optimizer_async",
                training_client.load_state_with_optimizer_async(checkpoint_state_path),
            )
            await self._tinker_train_call(
                "load_state_with_optimizer_result_async", load_future.result_async()
            )

        return training_client

    async def _save_checkpoint(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> tuple[Any, Any]:
        state_future, sampler_future = await asyncio.gather(
            self._tinker_train_call(
                "save_state_async",
                training_client.save_state_async(checkpoint_name),
            ),
            self._tinker_train_call(
                "save_weights_for_sampler_async",
                training_client.save_weights_for_sampler_async(checkpoint_name),
            ),
        )
        state_result = await self._tinker_train_call(
            "save_state_result_async", state_future.result_async()
        )
        sampler_result = await self._tinker_train_call(
            "save_weights_for_sampler_result_async", sampler_future.result_async()
        )
        return state_result, sampler_result

    async def _save_sampler_weights(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> Any:
        sampler_future = await self._tinker_train_call(
            "save_weights_for_sampler_async",
            training_client.save_weights_for_sampler_async(checkpoint_name),
        )
        return await self._tinker_train_call(
            "save_weights_for_sampler_result_async", sampler_future.result_async()
        )

    async def _save_training_state(
        self,
        training_client: tinker.TrainingClient,
        checkpoint_name: str,
    ) -> Any:
        state_future = await self._tinker_train_call(
            "save_state_async",
            training_client.save_state_async(checkpoint_name),
        )
        return await self._tinker_train_call(
            "save_state_result_async", state_future.result_async()
        )

    def _persist_model_state(self, model: TrainableModel, state: ModelState) -> None:
        model.merge_state(
            {
                STATE_KEY_RUN_IDS: state.tinker_run_ids,
                STATE_KEY_LATEST_STEP: state.current_step,
            }
        )

    async def _experimental_fork_checkpoint(
        self,
        model: Model,
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        """Fork a checkpoint from another TinkerNative model to initialize this model.

        Loads the source model's training checkpoint into the destination model's
        training client directly via tinker:// paths. No local download needed.

        Args:
            model: The destination model to fork to (must already be registered).
            from_model: The name of the source model to fork from.
            from_project: The project of the source model. Defaults to model.project.
            from_s3_bucket: Not supported for TinkerNativeBackend.
            not_after_step: If provided, uses the latest checkpoint <= this step.
            verbose: Whether to print verbose output.
            prefix: Not applicable for TinkerNativeBackend.
        """
        if from_s3_bucket is not None:
            raise NotImplementedError(
                "from_s3_bucket is not supported for TinkerNativeBackend. "
                "Tinker checkpoints are stored on Tinker infrastructure, not S3."
            )

        trainable_model = cast(TrainableModel, model)

        if trainable_model.name not in self._model_state:
            raise RuntimeError(
                f"Model '{trainable_model.name}' is not registered. "
                "Call register() before forking."
            )

        from_project = from_project or model.project

        # Read the source model's state.json to get its tinker_run_ids
        source_state_dir = get_model_dir(
            Model(name=from_model, project=from_project),
            art_path=self._path,
        )
        source_state_path = f"{source_state_dir}/state.json"
        import json

        if not os.path.exists(source_state_path):
            raise FileNotFoundError(
                f"Source model state not found at {source_state_path}. "
                f"Ensure the source model '{from_model}' has been trained "
                f"with this backend."
            )
        with open(source_state_path, "r") as f:
            source_state = json.load(f)

        source_run_ids = list(source_state.get(STATE_KEY_RUN_IDS, []))
        if not source_run_ids:
            raise ValueError(
                f"Source model '{from_model}' has no tinker run IDs in its state."
            )

        # List source model's checkpoints
        dest_state = self._model_state[trainable_model.name]
        training_paths, sampler_paths = await self._list_checkpoints(
            dest_state.rest_client, source_run_ids
        )

        if not training_paths:
            raise ValueError(
                f"No training checkpoints found for source model '{from_model}'."
            )

        # Select the target step
        available_steps = sorted(training_paths.keys())
        if not_after_step is not None:
            eligible_steps = [s for s in available_steps if s <= not_after_step]
            if not eligible_steps:
                raise ValueError(
                    f"No checkpoint found at or before step {not_after_step}. "
                    f"Available steps: {available_steps}"
                )
            target_step = max(eligible_steps)
        else:
            target_step = max(available_steps)

        source_checkpoint_path = training_paths[target_step]
        if verbose:
            print(
                f"Forking from '{from_model}' step {target_step} "
                f"(checkpoint: {source_checkpoint_path})"
            )

        # Load the source checkpoint into a new training client
        config = self._resolve_model_config(trainable_model)
        new_training_client = await self._create_training_client_from_checkpoint(
            service_client=dest_state.service_client,
            checkpoint_state_path=source_checkpoint_path,
            base_model=trainable_model.base_model,
            training_client_args=config.training_client_args,
            reset_optimizer=True,
        )

        # Save new sampler weights
        checkpoint_name = f"step_{target_step:06d}"
        sampler_response = await self._save_sampler_weights(
            new_training_client, checkpoint_name
        )

        # Create a sampler client from the new weights
        sampler_client = await self._tinker_train_call(
            "create_sampling_client_async",
            new_training_client.create_sampling_client_async(
                model_path=sampler_response.path
            ),
        )

        # Update the destination model's state
        new_run_id = new_training_client.model_id
        if new_run_id not in dest_state.tinker_run_ids:
            dest_state.tinker_run_ids.append(new_run_id)

        dest_state.training_client = new_training_client
        dest_state.current_step = target_step
        dest_state.sampler_clients[target_step] = sampler_client
        dest_state.sampler_checkpoint_paths[target_step] = sampler_response.path
        dest_state.training_checkpoint_paths[target_step] = source_checkpoint_path

        self._persist_model_state(trainable_model, dest_state)

        if verbose:
            print(
                f"Fork complete. Model '{trainable_model.name}' is now at "
                f"step {target_step}."
            )
