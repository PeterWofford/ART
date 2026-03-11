import importlib
import json

import pytest

from art.metrics import MetricsBuilder

ruler_module = importlib.import_module("art.rewards.ruler")


class _FakePromptTokenDetails:
    def __init__(self, *, cached_tokens: int = 0) -> None:
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakePromptTokenDetails(
            cached_tokens=cached_tokens
        )
        self.cost = cost
        self.model_extra = model_extra


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(
        self,
        *,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float | None = None,
        model_extra: dict[str, float] | None = None,
    ) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            model_extra=model_extra,
        )


@pytest.mark.asyncio
async def test_ruler_records_builder_cost_for_supported_judges(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Best answer.",
                            "score": 0.9,
                        }
                    ]
                }
            ),
            prompt_tokens=100,
            completion_tokens=50,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="openai/gpt-4.1",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.9)
    assert metrics["costs/train/judge/ruler"] == pytest.approx(0.0006)
    assert metrics["costs/train/judge"] == pytest.approx(0.0006)
    assert metrics["costs/train"] == pytest.approx(0.0006)
    assert metrics["costs/all"] == pytest.approx(0.0006)
    assert metrics["costs/cum/train/judge/ruler"] == pytest.approx(0.0006)


@pytest.mark.asyncio
async def test_ruler_skips_cost_when_pricing_is_unavailable(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Good enough.",
                            "score": 0.7,
                        }
                    ]
                }
            ),
            prompt_tokens=80,
            completion_tokens=20,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="ollama/qwen3:32b",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.7)
    assert not any(key.startswith("costs/") for key in metrics)


@pytest.mark.asyncio
async def test_ruler_records_direct_cost_for_openrouter_judges(monkeypatch):
    async def _fake_acompletion(**_kwargs):
        return _FakeResponse(
            content=json.dumps(
                {
                    "scores": [
                        {
                            "trajectory_id": "1",
                            "explanation": "Good enough.",
                            "score": 0.8,
                        }
                    ]
                }
            ),
            prompt_tokens=80,
            completion_tokens=20,
            cost=1.68e-05,
        )

    monkeypatch.setattr(ruler_module, "acompletion", _fake_acompletion)
    monkeypatch.setattr(ruler_module, "ModelResponse", _FakeResponse)

    builder = MetricsBuilder(cost_context="train")
    token = builder.activate()
    try:
        scores = await ruler_module.ruler(
            [[{"role": "user", "content": "test"}]],
            judge_model="openrouter/openai/gpt-4.1-mini",
        )
    finally:
        token.var.reset(token)

    metrics = await builder.flush()

    assert scores[0].score == pytest.approx(0.8)
    assert metrics["costs/train/judge/ruler"] == pytest.approx(1.68e-05)
