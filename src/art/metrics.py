from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction
import time
from typing import Any, ParamSpec, TypeVar

from .costs import tokens_to_cost

_active_builder: ContextVar["MetricsBuilder"] = ContextVar("_active_metrics_builder")

_HIERARCHICAL_SECTIONS = {"costs", "time", "data"}
_THROUGHPUT_IDLE_MAPPINGS = {
    "throughput/step_trainer_idle_s": "throughput/cum_trainer_idle_s",
    "throughput/step_actor_idle_s": "throughput/cum_actor_idle_s",
}
_DEFAULT_PROVIDER = "openai"
_OPENAI_PROVIDER = "openai"
_ANTHROPIC_PROVIDER = "anthropic"

P = ParamSpec("P")
R = TypeVar("R")


CostExtractor = Callable[[Any], float | None]
ResponseGetter = Callable[[Any], Any]


@dataclass(frozen=True)
class TokenPricing:
    prompt_per_million: float
    completion_per_million: float


_DEFAULT_TOKEN_PRICING = {
    _OPENAI_PROVIDER: TokenPricing(prompt_per_million=2.5, completion_per_million=10.0),
    _ANTHROPIC_PROVIDER: TokenPricing(
        prompt_per_million=3.0, completion_per_million=15.0
    ),
}


@dataclass
class _SharedMetricsState:
    lock: asyncio.Lock
    step_buffer: dict[str, float]
    cum_state: dict[str, float]
    unique_scenario_ids: set[str]
    pending_scenario_ids: set[str]
    cost_extractors: dict[str, CostExtractor]
    token_pricing: dict[str, TokenPricing]


def _new_shared_metrics_state() -> _SharedMetricsState:
    return _SharedMetricsState(
        lock=asyncio.Lock(),
        step_buffer={},
        cum_state={},
        unique_scenario_ids=set(),
        pending_scenario_ids=set(),
        cost_extractors={},
        token_pricing=dict(_DEFAULT_TOKEN_PRICING),
    )


def _normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    return normalized


def _read_usage_field(usage: Any, field: str) -> float | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    if value is None:
        return None
    return float(value)


def _response_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_openai_token_counts(response: Any) -> tuple[float, float] | None:
    usage = _response_usage(response)
    prompt_tokens = _read_usage_field(usage, "prompt_tokens")
    completion_tokens = _read_usage_field(usage, "completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return prompt_tokens or 0.0, completion_tokens or 0.0


def _extract_anthropic_token_counts(response: Any) -> tuple[float, float] | None:
    usage = _response_usage(response)
    input_tokens = _read_usage_field(usage, "input_tokens")
    output_tokens = _read_usage_field(usage, "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return input_tokens or 0.0, output_tokens or 0.0


def _detect_provider(response: Any) -> str | None:
    usage = _response_usage(response)
    if usage is None:
        return None

    if (
        _read_usage_field(usage, "prompt_tokens") is not None
        or _read_usage_field(usage, "completion_tokens") is not None
    ):
        return _OPENAI_PROVIDER
    if (
        _read_usage_field(usage, "input_tokens") is not None
        or _read_usage_field(usage, "output_tokens") is not None
    ):
        return _ANTHROPIC_PROVIDER
    return None


def _estimate_cost(
    token_counts: tuple[float, float] | None,
    pricing: TokenPricing,
) -> float | None:
    if token_counts is None:
        return None
    prompt_tokens, completion_tokens = token_counts
    return tokens_to_cost(prompt_tokens, pricing.prompt_per_million) + tokens_to_cost(
        completion_tokens,
        pricing.completion_per_million,
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
        self._lock = self._shared_state.lock
        self._step_buffer = self._shared_state.step_buffer
        self._cum_state = self._shared_state.cum_state
        self._unique_scenario_ids = self._shared_state.unique_scenario_ids
        self._pending_scenario_ids = self._shared_state.pending_scenario_ids
        self._cost_extractors = self._shared_state.cost_extractors
        self._token_pricing = self._shared_state.token_pricing

    def add_cost(self, path: str, usd: float) -> None:
        if not path:
            raise ValueError("Cost path must be non-empty")
        full_key = f"costs/{path}"
        self.add_metric(full_key, float(usd))

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
            self._pending_scenario_ids.update(
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

            if self._pending_scenario_ids:
                self._unique_scenario_ids.update(self._pending_scenario_ids)
                result["data/cum_num_unique_scenarios"] = float(
                    len(self._unique_scenario_ids)
                )

            self._update_throughput_metrics(result)
            self._step_buffer.clear()
            self._pending_scenario_ids.clear()
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
        normalized_provider = _normalize_provider(provider)
        if normalized_provider is None:
            raise ValueError("provider must be non-empty")
        self._cost_extractors[normalized_provider] = extractor

    def register_token_pricing(
        self,
        provider: str,
        *,
        prompt_per_million: float,
        completion_per_million: float,
    ) -> None:
        normalized_provider = _normalize_provider(provider)
        if normalized_provider is None:
            raise ValueError("provider must be non-empty")
        self._token_pricing[normalized_provider] = TokenPricing(
            prompt_per_million=float(prompt_per_million),
            completion_per_million=float(completion_per_million),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "cum_state": dict(self._cum_state),
            "unique_scenario_ids": list(self._unique_scenario_ids),
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

        # Keep local references aligned with the shared state so derived builders
        # created before or after resume observe the same cumulative state.
        self._cum_state = self._shared_state.cum_state
        self._unique_scenario_ids = self._shared_state.unique_scenario_ids
        self._pending_scenario_ids = self._shared_state.pending_scenario_ids

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

    def _update_throughput_metrics(self, result: dict[str, float]) -> None:
        for step_key, cum_key in _THROUGHPUT_IDLE_MAPPINGS.items():
            if step_key not in result:
                continue
            next_value = self._cum_state.get(cum_key, 0.0) + result[step_key]
            self._cum_state[cum_key] = next_value
            result[cum_key] = next_value

        if (
            "data/step_trainer_tokens" in result
            or "time/step_trainer_s" in result
        ):
            trainer_tokens = self._cum_state.get("data/step_trainer_tokens_cum")
            trainer_seconds = self._cum_state.get("time/step_trainer_s_cum")
            if (
                trainer_tokens is not None
                and trainer_seconds is not None
                and trainer_seconds > 0
            ):
                result["throughput/avg_trainer_tok_per_s"] = (
                    trainer_tokens / trainer_seconds
                )

        if "data/step_actor_tokens" in result or "time/step_actor_s" in result:
            actor_tokens = self._cum_state.get("data/step_actor_tokens_cum")
            actor_seconds = self._cum_state.get("time/step_actor_s_cum")
            if (
                actor_tokens is not None
                and actor_seconds is not None
                and actor_seconds > 0
            ):
                result["throughput/avg_actor_tok_per_s"] = actor_tokens / actor_seconds

    def _resolve_token_pricing(
        self,
        provider: str | None,
        *,
        prompt_price_per_million: float | None = None,
        completion_price_per_million: float | None = None,
    ) -> TokenPricing:
        normalized_provider = _normalize_provider(provider) or _DEFAULT_PROVIDER
        default_pricing = self._token_pricing.get(
            normalized_provider,
            self._token_pricing[_DEFAULT_PROVIDER],
        )
        return TokenPricing(
            prompt_per_million=(
                float(prompt_price_per_million)
                if prompt_price_per_million is not None
                else default_pricing.prompt_per_million
            ),
            completion_per_million=(
                float(completion_price_per_million)
                if completion_price_per_million is not None
                else default_pricing.completion_per_million
            ),
        )

    def _extract_api_cost(
        self,
        response: Any,
        *,
        provider: str | None = None,
        prompt_price_per_million: float | None = None,
        completion_price_per_million: float | None = None,
    ) -> float | None:
        provider_name = _normalize_provider(provider) or _detect_provider(response)
        if provider_name is not None:
            custom_extractor = self._cost_extractors.get(provider_name)
            if custom_extractor is not None:
                custom_cost = custom_extractor(response)
                if custom_cost is not None:
                    return float(custom_cost)

            token_pricing = self._resolve_token_pricing(
                provider_name,
                prompt_price_per_million=prompt_price_per_million,
                completion_price_per_million=completion_price_per_million,
            )
            if provider_name == _OPENAI_PROVIDER:
                return _estimate_cost(
                    _extract_openai_token_counts(response),
                    token_pricing,
                )
            if provider_name == _ANTHROPIC_PROVIDER:
                return _estimate_cost(
                    _extract_anthropic_token_counts(response),
                    token_pricing,
                )

        token_pricing = self._resolve_token_pricing(
            provider_name,
            prompt_price_per_million=prompt_price_per_million,
            completion_price_per_million=completion_price_per_million,
        )
        token_counts = _extract_openai_token_counts(response)
        if token_counts is None:
            token_counts = _extract_anthropic_token_counts(response)
        return _estimate_cost(token_counts, token_pricing)


def _record_api_cost(
    *,
    result: Any,
    source: str,
    provider: str | None,
    response_getter: ResponseGetter | None,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
) -> None:
    try:
        builder = MetricsBuilder.get_active()
    except LookupError:
        return

    response = response_getter(result) if response_getter is not None else result
    cost = builder._extract_api_cost(
        response,
        provider=provider,
        prompt_price_per_million=prompt_price_per_million,
        completion_price_per_million=completion_price_per_million,
    )
    if cost is None:
        return
    builder.add_cost(f"{builder.cost_context}/{source}", cost)


def track_api_cost(
    *,
    source: str,
    provider: str | None = None,
    response_getter: ResponseGetter | None = None,
    prompt_price_per_million: float | None = None,
    completion_price_per_million: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    normalized_source = source.strip("/")
    if not normalized_source:
        raise ValueError("source must be non-empty")

    normalized_provider = _normalize_provider(provider)

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        if iscoroutinefunction(func):

            @wraps(func)
            async def _async_wrapper(*args: P.args, **kwargs: P.kwargs):
                result = await func(*args, **kwargs)
                _record_api_cost(
                    result=result,
                    source=normalized_source,
                    provider=normalized_provider,
                    response_getter=response_getter,
                    prompt_price_per_million=prompt_price_per_million,
                    completion_price_per_million=completion_price_per_million,
                )
                return result

            return _async_wrapper

        @wraps(func)
        def _sync_wrapper(*args: P.args, **kwargs: P.kwargs):
            result = func(*args, **kwargs)
            _record_api_cost(
                result=result,
                source=normalized_source,
                provider=normalized_provider,
                response_getter=response_getter,
                prompt_price_per_million=prompt_price_per_million,
                completion_price_per_million=completion_price_per_million,
            )
            return result

        return _sync_wrapper

    return _decorate
