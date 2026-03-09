import asyncio
from contextvars import Token
from datetime import datetime
import json
import os
import time
from typing import TYPE_CHECKING, Any, Generic, Iterable, Optional, cast, overload
import warnings

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
import polars as pl
from pydantic import BaseModel
from typing_extensions import Never, TypeVar

from . import dev
from .costs import CostCalculator
from .metrics import MetricsBuilder
from .metrics_taxonomy import (
    TRAIN_GRADIENT_STEPS_KEY,
    average_metric_samples,
    build_data_metrics_from_summary,
    build_train_metrics_from_summary,
    summarize_trajectory_groups,
)
from .trajectories import Trajectory, TrajectoryGroup
from .types import TrainConfig, TrainSFTConfig
from .utils.trajectory_logging import write_trajectory_groups_parquet

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from art.backend import Backend


ModelConfig = TypeVar("ModelConfig", bound=BaseModel | None)
StateType = TypeVar("StateType", bound=dict[str, Any], default=dict[str, Any])

COSTS_METRIC_PREFIX = "costs_"
COSTS_TOTAL_KEY = f"{COSTS_METRIC_PREFIX}total"
METRICS_BUILDER_STATE_KEY = "_metrics_builder_state"
BUILDER_CUMULATIVE_PREFIXES = ("time/step_", "data/step_", "throughput/step_")
METRIC_SECTIONS = frozenset(
    {
        "reward",
        "loss",
        "offpolicy",
        "pipeline",
        "throughput",
        "costs",
        "time",
        "data",
    }
)
METRIC_SPLITS = frozenset({"train", "val", "test"})


class Model(
    BaseModel,
    Generic[ModelConfig, StateType],
):
    """
    A model is an object that can be passed to your `rollout` function, and used
    to log completions. Additionally, a `TrainableModel`, which is a subclass of
    `Model`, can be used to train a model.

    The `Model` abstraction is useful for comparing prompted model performance
    to the performance of your trained models.

    You can instantiate a prompted model like so:

    ``python model = art.Model(
        name="gpt-4.1", project="my-project",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )
    ``

    Or, if you're pointing at OpenRouter:

    ``python model = art.Model(
        name="gemini-2.5-pro", project="my-project",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
        inference_model_name="google/gemini-2.5-pro-preview-03-25",
    )
    ``

    For trainable (`art.TrainableModel`) models the inference values will be
    populated automatically by `model.register(api)` so you generally don't need
    to think about them.
    """

    name: str
    project: str
    entity: str | None = None
    id: str | None = None
    run_id: str | None = None
    config: ModelConfig
    # Discriminator field for FastAPI serialization
    trainable: bool = False

    # --- Inference connection information (populated automatically for
    #     TrainableModel or set manually for prompted / comparison models) ---
    inference_api_key: str | None = None
    inference_base_url: str | None = None
    # If set, this will be used instead of `self.name` when calling the
    # inference endpoint.
    inference_model_name: str | None = None

    # --- Frontend logging configuration ---
    base_path: str = ".art"  # Same default as LocalBackend for backward compat
    report_metrics: list[str] | None = None  # None = default (wandb if key present)

    _backend: Optional["Backend"] = None
    _s3_bucket: str | None = None
    _s3_prefix: str | None = None
    _openai_client: AsyncOpenAI | None = None
    _wandb_run: Optional["Run"] = None  # Private, for lazy wandb initialization
    _wandb_defined_metrics: set[str]
    _run_start_time: float
    _metrics_builder: MetricsBuilder
    _metrics_builder_state_loaded: bool
    _cost_calculator: CostCalculator

    def __init__(
        self,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig | None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            entity=entity,
            config=config,
            inference_api_key=inference_api_key,
            inference_base_url=inference_base_url,
            inference_model_name=inference_model_name,
            base_path=base_path,
            report_metrics=report_metrics,
            **kwargs,
        )
        object.__setattr__(self, "_wandb_defined_metrics", set())
        object.__setattr__(self, "_run_start_time", time.time())
        object.__setattr__(self, "_metrics_builder", MetricsBuilder(cost_context="train"))
        object.__setattr__(self, "_metrics_builder_state_loaded", False)

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: None = None,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
    ) -> "Model[None, dict[str, Any]]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig,
        inference_api_key: str | None = None,
        inference_base_url: str | None = None,
        inference_model_name: str | None = None,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
    ) -> "Model[ModelConfig, dict[str, Any]]": ...

    def __new__(  # pyright: ignore[reportInconsistentOverload]
        cls,
        *args,
        **kwargs,
    ) -> "Model[ModelConfig, StateType]":
        return super().__new__(cls)

    def safe_model_dump(self, *args, **kwargs) -> dict:
        """
        Dump the model, but remove the config field to prevent serialization errors in the backend.
        """
        data = super().model_dump(*args, **kwargs)
        # remove config from dumped_model to prevent serialization errors
        data["config"] = None
        return data

    def backend(self) -> "Backend":
        if self._backend is None:
            raise ValueError(
                "Model is not registered with the Backend. You must call `model.register()` first."
            )
        return self._backend

    async def register(self, backend: "Backend") -> None:
        if self.config is not None:
            try:
                self.config.model_dump_json()  # ty:ignore[invalid-argument-type, possibly-missing-attribute]
            except Exception as e:
                raise ValueError(
                    "The model config cannot be serialized to JSON. Please ensure that all fields are JSON serializable and try again."
                ) from e

        self._backend = backend
        await self._backend.register(self)

    def openai_client(
        self,
    ) -> AsyncOpenAI:
        if self._openai_client is not None:
            return self._openai_client

        if self.inference_api_key is None or self.inference_base_url is None:
            if self.trainable:
                raise ValueError(
                    "OpenAI client not yet available on this trainable model. You must call `model.register()` first."
                )
            else:
                raise ValueError(
                    "In order to create an OpenAI client you must provide an `inference_api_key` and `inference_base_url`."
                )
        self._openai_client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=1200, connect=5.0),
                limits=httpx.Limits(
                    max_connections=100_000, max_keepalive_connections=100_000
                ),
            ),
        )
        return self._openai_client

    def litellm_completion_params(self, step: int | None = None) -> dict:
        """Return the parameters that should be sent to litellm.completion.

        Args:
            step: If provided, returns params for specific checkpoint using
                  the `name@step` convention. If None, returns params for
                  latest checkpoint (default, backwards compatible).
        """
        model_name = self.get_inference_name(step)
        if self.trainable:
            model_name = f"hosted_vllm/{model_name}"
        return {
            "model": model_name,
            "base_url": self.inference_base_url,
            "api_key": self.inference_api_key,
            "temperature": 1,  # Important for trainable models
        }

    # ------------------------------------------------------------------
    # Inference name helpers
    # ------------------------------------------------------------------

    def get_inference_name(self, step: int | None = None) -> str:
        """Return the name that should be sent to the inference endpoint.

        Args:
            step: If provided, returns name for specific checkpoint.
                  If None, returns name for latest/default checkpoint.

        Note:
            For TrainableModel with LocalBackend, vLLM serves LoRA adapters
            as `model.name@step`, so this always includes the step suffix.
            For ServerlessBackend, it uses W&B artifact naming conventions.
        """
        # If we have a registered backend with _model_inference_name, use it
        # This ensures proper step handling for each backend type
        if self._backend is not None and hasattr(
            self._backend, "_model_inference_name"
        ):
            return self._backend._model_inference_name(self, step=step)

        # Fallback for non-registered models or backends without the method
        base_name = self.inference_model_name or self.name
        if step is not None:
            return f"{base_name}@{step}"
        return base_name

    def _get_output_dir(self) -> str:
        """Get the output directory for this model."""
        return f"{self.base_path}/{self.project}/models/{self.name}"

    def overwrite_state(self, state: StateType) -> None:
        """Overwrite persistent state in the model directory as JSON.

        This state is stored in `state.json` within the model's output directory
        and can be used to track training progress, dataset position, or any
        other information that should persist across runs.

        Warning:
            This overwrites the entire state file. Prefer `merge_state()` unless
            you intentionally want to replace all existing keys.

        Args:
            state: A dictionary of JSON-serializable values to persist.

        Example:
            model.overwrite_state({
                "step": 5,
                "dataset_offset": 100,
                "last_checkpoint_time": "2024-01-15T10:30:00",
            })
        """
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/state.json", "w") as f:
            json.dump(state, f, indent=2)

    def write_state(self, state: StateType) -> None:
        """Deprecated: use `overwrite_state()` or `merge_state()` instead."""
        warnings.warn(
            "write_state() is deprecated. Use overwrite_state() or merge_state() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.overwrite_state(state)

    def merge_state(self, state: StateType) -> StateType:
        """Deep-merge state into the existing state and persist it.

        Args:
            state: A dictionary of JSON-serializable values to merge.

        Returns:
            The merged state dictionary that was persisted.
        """
        existing = self.read_state() or {}
        merged = self._deep_merge_dicts(existing, state)
        self.overwrite_state(merged)
        return cast(StateType, merged)

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any], updates: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = Model._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def read_state(self) -> StateType | None:
        """Read persistent state from the model directory.

        Returns:
            The state dictionary if it exists, or None if no state has been saved.

        Example:
            state = model.read_state()
            if state:
                start_step = state["step"]
                dataset_offset = state["dataset_offset"]
        """
        output_dir = self._get_output_dir()
        state_path = f"{output_dir}/state.json"
        if not os.path.exists(state_path):
            return None
        with open(state_path, "r") as f:
            return json.load(f)

    def _get_wandb_run(self) -> Optional["Run"]:
        """Get or create the wandb run for this model."""
        import wandb

        if "WANDB_API_KEY" not in os.environ:
            return None
        if self._wandb_run is None or self._wandb_run._is_finished:
            run = wandb.init(
                project=self.project,
                name=self.name,
                id=self.name,
                resume="allow",
                settings=wandb.Settings(
                    x_stats_open_metrics_endpoints={
                        "vllm": "http://localhost:8000/metrics",
                    },
                    x_stats_open_metrics_filters=(
                        "vllm.vllm:num_requests_waiting",
                        "vllm.vllm:num_requests_running",
                    ),
                ),
            )
            self._wandb_run = run
            object.__setattr__(
                self,
                "_wandb_defined_metrics",
                {
                    "training_step",
                    "time/wall_clock_sec",
                },
            )

            # Define training_step as the x-axis for all metrics.
            # This allows out-of-order logging (e.g., async validation for previous steps).
            wandb.define_metric("training_step")
            wandb.define_metric("time/wall_clock_sec")
            wandb.define_metric("reward/*", step_metric="training_step")
            wandb.define_metric("loss/*", step_metric="training_step")
            wandb.define_metric("throughput/*", step_metric="training_step")
            wandb.define_metric("costs/*", step_metric="training_step")
            wandb.define_metric("time/*", step_metric="training_step")
            wandb.define_metric("data/*", step_metric="training_step")
            wandb.define_metric("train/*", step_metric="training_step")
            wandb.define_metric("val/*", step_metric="training_step")
            wandb.define_metric("test/*", step_metric="training_step")
        return self._wandb_run

    def _log_metrics(
        self,
        metrics: dict[str, float],
        split: str,
        step: int,
    ) -> None:
        """Log metrics to history.jsonl and optionally wandb."""
        if split in METRIC_SPLITS:
            prefixed = {}
            for key, value in metrics.items():
                first_component = key.split("/", 1)[0]
                has_prefix_component = "/" in key
                if has_prefix_component and (
                    first_component in METRIC_SECTIONS
                    or first_component in METRIC_SPLITS
                ):
                    prefixed[key] = value
                else:
                    prefixed[f"{split}/{key}"] = value
        else:
            prefixed = {f"{split}/{k}": v for k, v in metrics.items()}

        prefixed["training_step"] = step
        prefixed["time/wall_clock_sec"] = time.time() - self._run_start_time

        output_dir = self._get_output_dir()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write to history.jsonl
        with open(f"{output_dir}/history.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        k: v for k, v in prefixed.items() if v == v
                    }  # Filter out NaN values
                    | {"step": step, "recorded_at": datetime.now().isoformat()}
                )
                + "\n"
            )

        # Log to wandb if enabled
        should_log_wandb = (
            self.report_metrics is None and "WANDB_API_KEY" in os.environ
        ) or (self.report_metrics is not None and "wandb" in self.report_metrics)
        if should_log_wandb:
            if run := self._get_wandb_run():
                self._define_wandb_step_metrics(prefixed.keys())
                run.log(prefixed)

    def _define_wandb_step_metrics(self, keys: Iterable[str]) -> None:
        import wandb

        for key in keys:
            if not key.startswith("costs/"):
                continue
            if key in self._wandb_defined_metrics:
                continue
            wandb.define_metric(key, step_metric="training_step")
            self._wandb_defined_metrics.add(key)

    def _extract_non_cost_metrics(
        self, metrics: dict[str, float], split: str
    ) -> dict[str, float]:
        non_cost_metrics: dict[str, float] = {}
        cost_context = "train" if split == "train" else "eval"
        for metric, value in metrics.items():
            numeric_value = float(value)
            if metric == COSTS_TOTAL_KEY:
                raise ValueError(
                    "Do not log 'costs_total' directly. Log costs_* components "
                    "(e.g., costs_prefill, costs_sample) and totals are derived."
                )
            if metric.startswith("costs/"):
                self._metrics_builder.add_cost(metric[len("costs/") :], numeric_value)
                continue
            if metric.startswith(COSTS_METRIC_PREFIX):
                component = metric[len(COSTS_METRIC_PREFIX) :]
                if component:
                    self._metrics_builder.add_cost(
                        f"{cost_context}/{component}", numeric_value
                    )
                continue
            if metric.startswith(BUILDER_CUMULATIVE_PREFIXES):
                self._metrics_builder.add_metric(metric, numeric_value)
                continue
            non_cost_metrics[metric] = numeric_value
        return non_cost_metrics

    def _add_default_step_metrics(
        self,
        trajectory_groups: list[TrajectoryGroup],
        *,
        split: str,
        provided_metric_keys: set[str],
    ) -> dict[str, float]:
        if split not in METRIC_SPLITS:
            return {}

        summary = summarize_trajectory_groups(trajectory_groups)
        default_data_metrics = build_data_metrics_from_summary(
            summary,
            include_trainable_groups=split == "train",
        )
        for key, value in default_data_metrics.items():
            if key in provided_metric_keys:
                continue
            self._metrics_builder.add_metric(key, value)

        if summary.scenario_ids:
            self._metrics_builder.add_data(scenario_ids=summary.scenario_ids)

        if split != "train":
            return {}

        default_train_metrics = build_train_metrics_from_summary(summary)
        return {
            key: value
            for key, value in default_train_metrics.items()
            if key not in provided_metric_keys
        }

    def metrics_builder(self, cost_context: str | None = None) -> MetricsBuilder:
        self._load_metrics_builder_state()
        if cost_context is None:
            return self._metrics_builder
        return self._metrics_builder.for_cost_context(cost_context)

    def activate_metrics_context(
        self, cost_context: str
    ) -> Token[MetricsBuilder]:
        return self.metrics_builder(cost_context).activate()

    def _load_metrics_builder_state(self) -> None:
        if self._metrics_builder_state_loaded:
            return
        state = self.read_state() or {}
        metrics_state = state.get(METRICS_BUILDER_STATE_KEY)
        if isinstance(metrics_state, dict):
            self._metrics_builder.load_state_dict(metrics_state)
        object.__setattr__(self, "_metrics_builder_state_loaded", True)

    def _persist_metrics_builder_state(self) -> None:
        self.merge_state(
            {METRICS_BUILDER_STATE_KEY: self._metrics_builder.state_dict()}
        )

    def _normalize_trajectory_groups(
        self,
        trajectories: Iterable[Trajectory | BaseException] | Iterable[TrajectoryGroup],
    ) -> list[TrajectoryGroup]:
        items = list(trajectories)
        if not items:
            return []

        if all(isinstance(item, TrajectoryGroup) for item in items):
            return cast(list[TrajectoryGroup], items)

        if all(isinstance(item, (Trajectory, BaseException)) for item in items):
            return [TrajectoryGroup(cast(Iterable[Trajectory | BaseException], items))]

        raise TypeError(
            "trajectories must be an iterable of TrajectoryGroup objects or "
            "an iterable of Trajectory/BaseException items"
        )

    async def log(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[TrajectoryGroup] | None
        ) = None,
        split: str = "val",
        *,
        metrics: dict[str, float] | None = None,
        step: int | None = None,
    ) -> None:
        """
        Log trajectories and/or metrics.

        Can be used in two ways:
        1. Log trajectories: `await model.log(trajectory_groups, split="train")`
        2. Log raw metrics: `await model.log(metrics={"loss": 0.5}, step=1)`
        3. Both: `await model.log(trajectory_groups, metrics=extra_metrics)`

        Args:
            trajectories: A batch of trajectories or trajectory groups. Optional if
                logging only metrics.
            split: The evaluation's split. Defaults to "val".
            metrics: Optional dict of metrics to log directly (e.g., training metrics
                from backend.train()).
            step: Optional step number for metrics. If not provided, uses current step.
        """
        # Determine the step to use
        if step is None:
            step = await self.get_step() if self.trainable else 0

        self._load_metrics_builder_state()

        # If only metrics provided (no trajectories), just log them and return
        if trajectories is None:
            if metrics is not None:
                metrics_without_costs = self._extract_non_cost_metrics(metrics, split)
                if metrics_without_costs:
                    self._log_metrics(metrics_without_costs, split, step)
                costs = await self._metrics_builder.flush()
                if costs:
                    self._log_metrics(costs, split, step)
                self._persist_metrics_builder_state()
            return

        trajectory_groups = self._normalize_trajectory_groups(trajectories)

        default_train_metrics = self._add_default_step_metrics(
            trajectory_groups,
            split=split,
            provided_metric_keys=set(metrics or {}),
        )

        # Ensure output directories exist
        output_dir = self._get_output_dir()
        trajectories_dir = f"{output_dir}/trajectories/{split}"
        os.makedirs(trajectories_dir, exist_ok=True)

        # 1. Write parquet
        file_name = f"{step:04d}.parquet"
        write_trajectory_groups_parquet(
            trajectory_groups, f"{trajectories_dir}/{file_name}"
        )

        # 2. Calculate aggregate metrics (excluding additive costs)
        reward_key = "reward/mean" if split == "train" else "reward"
        exception_rate_key = (
            "reward/exception_rate" if split == "train" else "exception_rate"
        )
        reward_std_dev_key = "reward/std_dev" if split == "train" else "reward_std_dev"

        all_metrics: dict[str, list[float]] = {
            reward_key: [],
            exception_rate_key: [],
        }
        group_metrics: dict[str, list[float]] = {}

        for group in trajectory_groups:
            if group.metrics:
                group_non_cost = self._extract_non_cost_metrics(
                    cast(dict[str, float], group.metrics), split
                )
            else:
                group_non_cost = {}
            if group.trajectories:
                for metric, value in group_non_cost.items():
                    if metric not in group_metrics:
                        group_metrics[metric] = []
                    group_metrics[metric].append(float(value))

            all_metrics[exception_rate_key].extend(0.0 for _ in group.trajectories)
            all_metrics[exception_rate_key].extend(1.0 for _ in group.exceptions)

            for trajectory in group.trajectories:
                all_metrics[reward_key].append(trajectory.reward)

                # Collect other custom metrics
                trajectory_metrics: dict[str, float] = {}
                for metric, value in trajectory.metrics.items():
                    routed_metric = metric
                    if split == "train" and "/" not in routed_metric:
                        routed_metric = f"reward/{routed_metric}"
                    trajectory_metrics[routed_metric] = float(value)

                non_cost_trajectory_metrics = self._extract_non_cost_metrics(
                    trajectory_metrics,
                    split,
                )
                for metric, value in non_cost_trajectory_metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))

        # Calculate averages for all metrics
        averages: dict[str, float] = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        averages.update(default_train_metrics)

        # Aggregate group-level metrics once per group
        for metric, values in group_metrics.items():
            if len(values) > 0:
                group_key = (
                    f"reward/group_{metric}" if split == "train" else f"group_metric_{metric}"
                )
                averages[group_key] = sum(values) / len(values)

        # Calculate average standard deviation of rewards within groups
        from .utils.old_benchmarking.calculate_step_metrics import (
            calculate_step_std_dev,
        )

        averages[reward_std_dev_key] = calculate_step_std_dev(trajectory_groups)

        # Merge in any additional metrics passed directly
        if metrics is not None:
            metrics_without_costs = self._extract_non_cost_metrics(metrics, split)
            averages.update(metrics_without_costs)

        # 3. Log metrics (writes to history.jsonl and wandb)
        self._log_metrics(averages, split, step)

        # 4. Log cumulative costs
        costs = await self._metrics_builder.flush()
        if costs:
            self._log_metrics(costs, split, step)
        self._persist_metrics_builder_state()

    async def get_step(self) -> int:
        """
        Get the model's current training step. For non-trainable models, returns 0.
        """
        if self.trainable:
            return await self.backend()._get_step(self)  # type: ignore
        return 0


# ---------------------------------------------------------------------------
# Trainable models
# ---------------------------------------------------------------------------


class TrainableModel(Model[ModelConfig, StateType], Generic[ModelConfig, StateType]):
    base_model: str
    # Override discriminator field for FastAPI serialization
    trainable: bool = True

    # The fields within `_internal_config` are unstable and subject to change.
    # Use at your own risk.
    _internal_config: dev.InternalModelConfig | None = None

    def __init__(
        self,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        run_id: str | None = None,
        config: ModelConfig | None = None,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
        **kwargs: Never,
    ) -> None:
        super().__init__(
            name=name,
            project=project,
            entity=entity,
            id=id,
            config=config,
            base_model=base_model,
            base_path=base_path,
            report_metrics=report_metrics,
            **kwargs,
        )
        object.__setattr__(self, "_cost_calculator", self._noop_cost_calculator)
        if _internal_config is not None:
            # Bypass BaseModel __setattr__ to allow setting private attr
            object.__setattr__(self, "_internal_config", _internal_config)

    @property
    def cost_calculator(self) -> CostCalculator:
        return self._cost_calculator

    def set_cost_calculator(self, calculator: CostCalculator | None) -> None:
        object.__setattr__(
            self,
            "_cost_calculator",
            calculator if calculator is not None else self._noop_cost_calculator,
        )

    @staticmethod
    def _noop_cost_calculator(
        _prompt_tokens: int | None, _completion_tokens: int | None
    ) -> dict[str, float]:
        return {}

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: None = None,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[None, dict[str, Any]]": ...

    @overload
    def __new__(
        cls,
        *,
        name: str,
        project: str,
        entity: str | None = None,
        id: str | None = None,
        config: ModelConfig,
        base_model: str,
        base_path: str = ".art",
        report_metrics: list[str] | None = None,
        _internal_config: dev.InternalModelConfig | None = None,
    ) -> "TrainableModel[ModelConfig, dict[str, Any]]": ...

    def __new__(  # pyright: ignore[reportInconsistentOverload]
        cls,
        *args,
        **kwargs,
    ) -> "TrainableModel[ModelConfig, StateType]":
        return super().__new__(cls)

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        data["_internal_config"] = self._internal_config
        return data

    def safe_model_dump(self, *args, **kwargs) -> dict:
        """
        Dump the model, but remove the config field to prevent serialization errors in the backend.
        """
        data = self.model_dump(*args, **kwargs)
        # remove config from dumped_model to prevent serialization errors
        data["config"] = None
        return data

    async def register(
        self,
        backend: "Backend",
        _openai_client_config: dev.OpenAIServerConfig | None = None,
    ) -> None:
        await super().register(backend)
        base_url, api_key = await backend._prepare_backend_for_training(
            self, _openai_client_config
        )

        # Populate the top-level inference fields so that the rest of the
        # code (and any user code) can create an OpenAI client immediately.
        self.inference_base_url = base_url
        self.inference_api_key = api_key
        self.inference_model_name = (
            hasattr(backend, "_model_inference_name")
            and getattr(backend, "_model_inference_name")(self)
            or self.name
        )

    async def delete_checkpoints(
        self, best_checkpoint_metric: str = "val/reward"
    ) -> None:
        """
        Delete all but the latest and best checkpoints.

        Args:
            best_checkpoint_metric: The metric to use to determine the best checkpoint.
                Defaults to "val/reward".
        """
        output_dir = self._get_output_dir()
        steps_to_keep = [await self.get_step()]  # Keep latest

        # Read history.jsonl to find best step
        try:
            best_step = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=[best_checkpoint_metric])
                .group_by("step")
                .mean()
                .sort(best_checkpoint_metric)
                .select(pl.col("step").last())
                .item()
            )
            steps_to_keep.append(best_step)
        except FileNotFoundError:
            print(f'"{output_dir}/history.jsonl" not found')
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "{best_checkpoint_metric}" metric found in history')

        # Backend only does file deletion
        await self.backend()._delete_checkpoint_files(self, steps_to_keep)

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        config: TrainConfig = TrainConfig(),
        _config: dev.TrainConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Reinforce fine-tune the model with a batch of trajectory groups.

        .. deprecated::
            Use ``backend.train(model, trajectory_groups, ...)`` instead.
            This method will be removed in a future version.

        Args:
            trajectory_groups: A batch of trajectory groups.
            config: Fine-tuning specific configuration
            _config: Additional configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
        """
        warnings.warn(
            "model.train() is deprecated. Use backend.train(model, ...) instead.\n\n"
            "Migration guide:\n"
            "  # Before (deprecated):\n"
            "  await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))\n\n"
            "  # After (recommended):\n"
            "  result = await backend.train(model, trajectory_groups, learning_rate=5e-6)\n"
            "  await model.log(trajectory_groups, metrics=result.metrics, step=result.step, split='train')\n\n"
            "Key differences:\n"
            "  - backend.train() does NOT automatically log trajectories or metrics\n"
            "  - backend.train() returns a TrainResult with step, metrics, and checkpoint info\n"
            "  - Each backend has its own type-checked parameters (no more generic config objects)",
            DeprecationWarning,
            stacklevel=2,
        )
        groups_list = list(trajectory_groups)
        _config = _config or {}  # ty:ignore[invalid-assignment]

        # 1. Train (backend no longer logs internally)
        training_metrics: list[dict[str, float]] = []
        trainer_started = time.monotonic()
        async for metrics in self.backend()._train_model(
            self,
            groups_list,
            config,
            _config,  # ty:ignore[invalid-argument-type]
            verbose,
        ):
            training_metrics.append(metrics)
        trainer_elapsed = time.monotonic() - trainer_started

        # 2. Calculate aggregated training metrics
        avg_metrics = average_metric_samples(training_metrics)
        avg_metrics.setdefault("time/step_trainer_s", trainer_elapsed)

        # 3. Log trajectories and training metrics together (single wandb log call)
        step = await self.get_step()
        await self.log(groups_list, split="train", metrics=avg_metrics, step=step)

    async def train_sft(
        self,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig | None = None,
        _config: dev.TrainSFTConfig | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Supervised fine-tune the model with an iterable of trajectories.

        Args:
            trajectories: An iterable of Trajectory objects.
            config: SFT configuration including learning_rates and batch_size.
                If None, uses default TrainSFTConfig().
            _config: Additional experimental configuration that is subject to change and
                not yet part of the public API. Use at your own risk.
            verbose: Whether to print verbose output.
        """
        if config is None:
            config = TrainSFTConfig()

        # Train (backend yields metrics for each batch without logging)
        # Collect all metrics and aggregate them at the end (same as RL)
        _config = _config or {}  # ty:ignore[invalid-assignment]
        training_metrics: list[dict[str, float]] = []
        trainer_started = time.monotonic()
        async for metrics in self.backend()._train_sft(
            self,
            trajectories,
            config,
            _config,  # ty:ignore[invalid-argument-type]
            verbose,
        ):
            training_metrics.append(metrics)
        trainer_elapsed = time.monotonic() - trainer_started

        # Log aggregated training metrics once (same as RL)
        if training_metrics:
            avg_metrics = average_metric_samples(training_metrics)
            avg_metrics["time/step_trainer_s"] = trainer_elapsed
            # Get the current step after training
            step = await self.get_step()
            await self.log(trajectories=None, split="train", metrics=avg_metrics, step=step)
