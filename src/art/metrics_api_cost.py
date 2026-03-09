from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, ParamSpec, TypeVar

from .costs import tokens_to_cost

DEFAULT_PROVIDER = "openai"
OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"

P = ParamSpec("P")
R = TypeVar("R")

CostExtractor = Callable[[Any], float | None]
ResponseGetter = Callable[[Any], Any]


@dataclass(frozen=True)
class TokenPricing:
    prompt_per_million: float
    completion_per_million: float


DEFAULT_TOKEN_PRICING = {
    OPENAI_PROVIDER: TokenPricing(prompt_per_million=2.5, completion_per_million=10.0),
    ANTHROPIC_PROVIDER: TokenPricing(
        prompt_per_million=3.0,
        completion_per_million=15.0,
    ),
}


def normalize_provider(provider: str | None) -> str | None:
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
        return OPENAI_PROVIDER
    if (
        _read_usage_field(usage, "input_tokens") is not None
        or _read_usage_field(usage, "output_tokens") is not None
    ):
        return ANTHROPIC_PROVIDER
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


def _resolve_token_pricing(
    provider: str | None,
    *,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    token_pricing: Mapping[str, TokenPricing],
) -> TokenPricing:
    normalized_provider = normalize_provider(provider) or DEFAULT_PROVIDER
    default_pricing = token_pricing.get(
        normalized_provider,
        token_pricing[DEFAULT_PROVIDER],
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


def extract_api_cost(
    response: Any,
    *,
    provider: str | None,
    prompt_price_per_million: float | None,
    completion_price_per_million: float | None,
    cost_extractors: Mapping[str, CostExtractor],
    token_pricing: Mapping[str, TokenPricing],
) -> float | None:
    provider_name = normalize_provider(provider) or _detect_provider(response)
    if provider_name is not None:
        custom_extractor = cost_extractors.get(provider_name)
        if custom_extractor is not None:
            custom_cost = custom_extractor(response)
            if custom_cost is not None:
                return float(custom_cost)

        pricing = _resolve_token_pricing(
            provider_name,
            prompt_price_per_million=prompt_price_per_million,
            completion_price_per_million=completion_price_per_million,
            token_pricing=token_pricing,
        )
        if provider_name == OPENAI_PROVIDER:
            return _estimate_cost(_extract_openai_token_counts(response), pricing)
        if provider_name == ANTHROPIC_PROVIDER:
            return _estimate_cost(_extract_anthropic_token_counts(response), pricing)

    pricing = _resolve_token_pricing(
        provider_name,
        prompt_price_per_million=prompt_price_per_million,
        completion_price_per_million=completion_price_per_million,
        token_pricing=token_pricing,
    )
    token_counts = _extract_openai_token_counts(response)
    if token_counts is None:
        token_counts = _extract_anthropic_token_counts(response)
    return _estimate_cost(token_counts, pricing)


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
        from .metrics import MetricsBuilder

        builder = MetricsBuilder.get_active()
    except LookupError:
        return

    response = response_getter(result) if response_getter is not None else result
    builder.add_response_cost(
        source,
        response,
        provider=provider,
        prompt_price_per_million=prompt_price_per_million,
        completion_price_per_million=completion_price_per_million,
    )


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

    normalized_provider = normalize_provider(provider)

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
