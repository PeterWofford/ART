from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import time
from typing import Any

from .api_costs import (
    CostExtractor,
    ModelNameGetter,
    TokenPricing,
    extract_api_cost,
    normalize_provider,
)

_active_builder: ContextVar["MetricsBuilder"] = ContextVar("_active_metrics_builder")

_HIERARCHICAL_SECTIONS = {"costs", "time", "data"}
_THROUGHPUT_IDLE_MAPPINGS = {
    "throughput/step_trainer_idle_s": "throughput/cum/trainer_idle_s",
    "throughput/step_actor_idle_s": "throughput/cum/actor_idle_s",
}


def is_cumulative_metric_key(key: str) -> bool:
    parts = key.split("/", 2)
    return len(parts) >= 2 and parts[1] == "cum"


def is_builder_managed_metric(key: str) -> bool:
    return key.startswith(("costs/", "time/step_", "data/step_", "throughput/step_"))


def to_cumulative_metric_key(key: str) -> str:
    if is_cumulative_metric_key(key):
        raise ValueError(f"Metric key '{key}' is already cumulative.")

    section, rest = key.split("/", 1)
    if rest.startswith("step_"):
        rest = rest[len("step_") :]
    return f"{section}/cum/{rest}"


@dataclass
class _SharedMetricsState:
    lock: asyncio.Lock
    step_buffer: dict[str, float]
    cum_state: dict[str, float]
    unique_scenario_ids: set[str]
    pending_scenario_ids: set[str]
    cost_extractors: dict[str, CostExtractor]
    model_pricing: dict[str, TokenPricing]


def _new_shared_metrics_state() -> _SharedMetricsState:
    return _SharedMetricsState(
        lock=asyncio.Lock(),
        step_buffer={},
        cum_state={},
        unique_scenario_ids=set(),
        pending_scenario_ids=set(),
        cost_extractors={},
        model_pricing={},
    )


class MetricsBuilder:
    """Build and accumulate step-level metrics for logging."""

    def __init__(
        self,
        cost_context: str,
        *,
        _shared_state: _SharedMetricsState | None = None,
    ) -> None:
        if not cost_context:
            raise ValueError("cost_context must be non-empty")

        self.cost_context = cost_context
        self._shared_state = (
            _shared_state if _shared_state is not None else _new_shared_metrics_state()
        )

    def add_cost(self, path: str, usd: float) -> None:
        if not path:
            raise ValueError("Cost path must be non-empty")
        full_key = f"costs/{path}"
        self.add_metric(full_key, float(usd))

    def add_response_cost(
        self,
        source: str,
        response: Any,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        model_name_getter: "ModelNameGetter | None" = None,
        prompt_price_per_million: float | None = None,
        completion_price_per_million: float | None = None,
        cached_prompt_price_per_million: float | None = None,
        cache_creation_price_per_million: float | None = None,
        cache_read_price_per_million: float | None = None,
    ) -> float | None:
        normalized_source = source.strip("/")
        if not normalized_source:
            raise ValueError("source must be non-empty")

        cost = extract_api_cost(
            response,
            provider=provider,
            model_name=model_name,
            model_name_getter=model_name_getter,
            prompt_price_per_million=prompt_price_per_million,
            completion_price_per_million=completion_price_per_million,
            cached_prompt_price_per_million=cached_prompt_price_per_million,
            cache_creation_price_per_million=cache_creation_price_per_million,
            cache_read_price_per_million=cache_read_price_per_million,
            cost_extractors=self._shared_state.cost_extractors,
            model_pricing=self._shared_state.model_pricing,
        )
        if cost is None:
            return None

        self.add_cost(f"{self.cost_context}/{normalized_source}", cost)
        return cost

    def add_metric(self, key: str, value: float) -> None:
        if "/" not in key:
            raise ValueError("Metric key must include a section prefix")
        self._validate_and_add(key, float(value))

    def add_data(
        self,
        step_num_scenarios: int | None = None,
        step_actor_tokens: int | None = None,
        scenario_ids: list[str] | None = None,
    ) -> None:
        if step_num_scenarios is not None:
            self.add_metric("data/step_num_scenarios", float(step_num_scenarios))
        if step_actor_tokens is not None:
            self.add_metric("data/step_actor_tokens", float(step_actor_tokens))
        if scenario_ids is not None:
            self._shared_state.pending_scenario_ids.update(
                str(scenario_id) for scenario_id in scenario_ids
            )

    def add_user_timing(
        self,
        step_wall_s: float | None = None,
        step_actor_s: float | None = None,
        step_eval_s: float | None = None,
    ) -> None:
        if step_wall_s is not None:
            self.add_metric("time/step_wall_s", float(step_wall_s))
        if step_actor_s is not None:
            self.add_metric("time/step_actor_s", float(step_actor_s))
        if step_eval_s is not None:
            self.add_metric("time/step_eval_s", float(step_eval_s))

    def add_idle_times(
        self,
        step_trainer_idle_s: float | None = None,
        step_actor_idle_s: float | None = None,
    ) -> None:
        if step_trainer_idle_s is not None:
            self.add_metric(
                "throughput/step_trainer_idle_s",
                float(step_trainer_idle_s),
            )
        if step_actor_idle_s is not None:
            self.add_metric("throughput/step_actor_idle_s", float(step_actor_idle_s))

    @contextmanager
    def measure(self, key: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.add_metric(key, time.monotonic() - started)

    async def flush(self) -> dict[str, float]:
        async with self._shared_state.lock:
            self._validate_hierarchy()

            result = dict(self._shared_state.step_buffer)
            cost_metrics = {
                key: value
                for key, value in self._shared_state.step_buffer.items()
                if key.startswith("costs/")
            }
            result.update(self._compute_rollups(cost_metrics))

            for key, value in list(result.items()):
                section = key.split("/", 1)[0]
                if section not in _HIERARCHICAL_SECTIONS:
                    continue
                cum_key = to_cumulative_metric_key(key)
                next_value = self._shared_state.cum_state.get(cum_key, 0.0) + value
                self._shared_state.cum_state[cum_key] = next_value
                result[cum_key] = next_value

            if self._shared_state.pending_scenario_ids:
                self._shared_state.unique_scenario_ids.update(
                    self._shared_state.pending_scenario_ids
                )
                result["data/cum/num_unique_scenarios"] = float(
                    len(self._shared_state.unique_scenario_ids)
                )

            self._update_throughput_metrics(result)
            self._shared_state.step_buffer.clear()
            self._shared_state.pending_scenario_ids.clear()
            return result

    def activate(self) -> Token["MetricsBuilder"]:
        return _active_builder.set(self)

    @contextmanager
    def activate_context(self):
        token = self.activate()
        try:
            yield self
        finally:
            token.var.reset(token)

    @staticmethod
    def get_active() -> "MetricsBuilder":
        return _active_builder.get()

    def for_cost_context(self, cost_context: str) -> "MetricsBuilder":
        normalized_cost_context = cost_context.strip()
        if not normalized_cost_context:
            raise ValueError("cost_context must be non-empty")
        if normalized_cost_context == self.cost_context:
            return self
        return MetricsBuilder(
            cost_context=normalized_cost_context,
            _shared_state=self._shared_state,
        )

    def register_cost_extractor(
        self, provider: str, extractor: CostExtractor
    ) -> None:
        normalized_provider = normalize_provider(provider)
        if normalized_provider is None:
            raise ValueError("provider must be non-empty")
        self._shared_state.cost_extractors[normalized_provider] = extractor

    def register_model_pricing(
        self,
        model_name: str,
        *,
        prompt_per_million: float,
        completion_per_million: float,
        cached_prompt_per_million: float | None = None,
        cache_creation_per_million: float | None = None,
        cache_read_per_million: float | None = None,
    ) -> None:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must be non-empty")
        self._shared_state.model_pricing[normalized_model_name] = TokenPricing(
            prompt_per_million=float(prompt_per_million),
            completion_per_million=float(completion_per_million),
            cached_prompt_per_million=(
                float(cached_prompt_per_million)
                if cached_prompt_per_million is not None
                else None
            ),
            cache_creation_per_million=(
                float(cache_creation_per_million)
                if cache_creation_per_million is not None
                else None
            ),
            cache_read_per_million=(
                float(cache_read_per_million)
                if cache_read_per_million is not None
                else None
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "cum_state": dict(self._shared_state.cum_state),
            "unique_scenario_ids": list(self._shared_state.unique_scenario_ids),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raw_cum_state = state.get("cum_state", {})
        raw_unique_ids = state.get("unique_scenario_ids", [])
        restored_cum_state = {str(k): float(v) for k, v in raw_cum_state.items()}
        restored_unique_ids = {str(v) for v in raw_unique_ids}

        self._shared_state.cum_state.clear()
        self._shared_state.cum_state.update(restored_cum_state)
        self._shared_state.unique_scenario_ids.clear()
        self._shared_state.unique_scenario_ids.update(restored_unique_ids)
        self._shared_state.pending_scenario_ids.clear()

    def _validate_and_add(self, key: str, value: float) -> None:
        if is_cumulative_metric_key(key):
            raise ValueError(
                f"Metric key '{key}' uses the reserved cumulative namespace."
            )

        for existing_key in self._shared_state.step_buffer:
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

        self._shared_state.step_buffer[key] = (
            self._shared_state.step_buffer.get(key, 0.0) + value
        )

    def _validate_hierarchy(self) -> None:
        keys = sorted(
            k for k in self._shared_state.step_buffer if k.startswith("costs/")
        )
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

    def _update_throughput_metrics(self, result: dict[str, float]) -> None:
        for step_key, cum_key in _THROUGHPUT_IDLE_MAPPINGS.items():
            if step_key not in result:
                continue
            next_value = (
                self._shared_state.cum_state.get(cum_key, 0.0) + result[step_key]
            )
            self._shared_state.cum_state[cum_key] = next_value
            result[cum_key] = next_value

        if (
            "data/step_trainer_tokens" in result
            or "time/step_trainer_s" in result
        ):
            trainer_tokens = self._shared_state.cum_state.get(
                "data/cum/trainer_tokens"
            )
            trainer_seconds = self._shared_state.cum_state.get(
                "time/cum/trainer_s"
            )
            if (
                trainer_tokens is not None
                and trainer_seconds is not None
                and trainer_seconds > 0
            ):
                result["throughput/avg_trainer_tok_per_s"] = (
                    trainer_tokens / trainer_seconds
                )

        if "data/step_actor_tokens" in result or "time/step_actor_s" in result:
            actor_tokens = self._shared_state.cum_state.get(
                "data/cum/actor_tokens"
            )
            actor_seconds = self._shared_state.cum_state.get("time/cum/actor_s")
            if (
                actor_tokens is not None
                and actor_seconds is not None
                and actor_seconds > 0
            ):
                result["throughput/avg_actor_tok_per_s"] = actor_tokens / actor_seconds


from .api_costs import track_api_cost
