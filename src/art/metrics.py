from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Any

_active_builder: ContextVar["MetricsBuilder"] = ContextVar("_active_metrics_builder")

_HIERARCHICAL_SECTIONS = {"costs", "time", "data"}


class MetricsBuilder:
    """Build and accumulate step-level metrics for logging."""

    def __init__(self, cost_context: str) -> None:
        if not cost_context:
            raise ValueError("cost_context must be non-empty")

        self.cost_context = cost_context
        self._lock = asyncio.Lock()
        self._step_buffer: dict[str, float] = {}
        self._cum_state: dict[str, float] = {}
        self._unique_scenario_ids: set[str] = set()

    def add_cost(self, path: str, usd: float) -> None:
        if not path:
            raise ValueError("Cost path must be non-empty")
        full_key = f"costs/{path}"
        self._validate_and_add(full_key, float(usd))

    def add_data(
        self,
        step_num_scenarios: int | None = None,
        step_actor_tokens: int | None = None,
        scenario_ids: list[str] | None = None,
    ) -> None:
        if step_num_scenarios is not None:
            self._step_buffer["data/step_num_scenarios"] = float(step_num_scenarios)
        if step_actor_tokens is not None:
            self._step_buffer["data/step_actor_tokens"] = float(step_actor_tokens)
        if scenario_ids is not None:
            self._unique_scenario_ids.update(scenario_ids)

    def add_user_timing(
        self,
        step_wall_s: float | None = None,
        step_actor_s: float | None = None,
        step_eval_s: float | None = None,
    ) -> None:
        if step_wall_s is not None:
            self._step_buffer["time/step_wall_s"] = float(step_wall_s)
        if step_actor_s is not None:
            self._step_buffer["time/step_actor_s"] = float(step_actor_s)
        if step_eval_s is not None:
            self._step_buffer["time/step_eval_s"] = float(step_eval_s)

    def add_idle_times(
        self,
        step_trainer_idle_s: float | None = None,
        step_actor_idle_s: float | None = None,
    ) -> None:
        if step_trainer_idle_s is not None:
            self._step_buffer["throughput/step_trainer_idle_s"] = float(
                step_trainer_idle_s
            )
        if step_actor_idle_s is not None:
            self._step_buffer["throughput/step_actor_idle_s"] = float(step_actor_idle_s)

    async def flush(self, step: int) -> dict[str, float]:
        del step
        async with self._lock:
            self._validate_hierarchy()

            result = dict(self._step_buffer)
            cost_metrics = {
                key: value
                for key, value in self._step_buffer.items()
                if key.startswith("costs/")
            }
            result.update(self._compute_rollups(cost_metrics))

            for key, value in list(result.items()):
                section = key.split("/", 1)[0]
                if section not in _HIERARCHICAL_SECTIONS:
                    continue
                cum_key = f"{key}_cum"
                next_value = self._cum_state.get(cum_key, 0.0) + value
                self._cum_state[cum_key] = next_value
                result[cum_key] = next_value

            if self._unique_scenario_ids:
                result["data/cum_num_unique_scenarios"] = float(
                    len(self._unique_scenario_ids)
                )

            self._step_buffer.clear()
            return result

    def activate(self) -> Token["MetricsBuilder"]:
        return _active_builder.set(self)

    @staticmethod
    def get_active() -> "MetricsBuilder":
        return _active_builder.get()

    def state_dict(self) -> dict[str, Any]:
        return {
            "cum_state": dict(self._cum_state),
            "unique_scenario_ids": list(self._unique_scenario_ids),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raw_cum_state = state.get("cum_state", {})
        raw_unique_ids = state.get("unique_scenario_ids", [])
        self._cum_state = {str(k): float(v) for k, v in raw_cum_state.items()}
        self._unique_scenario_ids = {str(v) for v in raw_unique_ids}

    def _validate_and_add(self, key: str, value: float) -> None:
        if key.endswith("_cum"):
            raise ValueError(
                f"Metric key '{key}' ends with '_cum', which is reserved for cumulative metrics."
            )

        for existing_key in self._step_buffer:
            if existing_key == key:
                continue
            if existing_key.startswith(f"{key}/"):
                raise ValueError(
                    f"Cannot log '{key}' as a leaf: it is an ancestor of '{existing_key}'."
                )
            if key.startswith(f"{existing_key}/"):
                raise ValueError(
                    f"Cannot log '{key}' as a leaf: '{existing_key}' is already a leaf ancestor."
                )

        self._step_buffer[key] = self._step_buffer.get(key, 0.0) + value

    def _validate_hierarchy(self) -> None:
        keys = sorted(k for k in self._step_buffer if k.startswith("costs/"))
        for i, key in enumerate(keys):
            for other in keys[i + 1 :]:
                if other.startswith(f"{key}/"):
                    raise ValueError(
                        f"Leaf/parent conflict: '{key}' and '{other}' cannot coexist."
                    )

    def _compute_rollups(self, cost_metrics: dict[str, float]) -> dict[str, float]:
        if not cost_metrics:
            return {}

        all_parents: set[str] = set()
        for key in cost_metrics:
            parts = key.split("/")
            for depth in range(2, len(parts)):
                all_parents.add("/".join(parts[:depth]))

        rollups: dict[str, float] = {}
        for parent in all_parents:
            prefix = f"{parent}/"
            rollups[parent] = sum(
                value for key, value in cost_metrics.items() if key.startswith(prefix)
            )

        top_level_children = {key.split("/")[1] for key in cost_metrics}
        costs_all = 0.0
        for child_name in top_level_children:
            child_key = f"costs/{child_name}"
            if child_key in rollups:
                costs_all += rollups[child_key]
            else:
                costs_all += cost_metrics[child_key]
        rollups["costs/all"] = costs_all

        return rollups
