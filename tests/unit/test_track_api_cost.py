import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from art import Model, TrainableModel, Trajectory, TrajectoryGroup
from art.metrics import MetricsBuilder, track_api_cost
from art.pipeline_trainer.trainer import PipelineTrainer


class _OpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _OpenAIResponse:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        model: str | None = None,
    ) -> None:
        self.usage = _OpenAIUsage(prompt_tokens, completion_tokens)
        self.model = model


class _AnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _AnthropicResponse:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        model: str | None = None,
    ) -> None:
        self.usage = _AnthropicUsage(input_tokens, output_tokens)
        self.model = model


class TestTrackApiCost:
    @pytest.mark.asyncio
    async def test_openai_cost_extraction_with_explicit_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/correctness"] == pytest.approx(0.0002)

    @pytest.mark.asyncio
    async def test_anthropic_cost_extraction_uses_registered_model_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.register_model_pricing(
            "anthropic/test-judge",
            prompt_per_million=5.0,
            completion_per_million=7.0,
        )

        @track_api_cost(
            source="llm_judge/faithfulness",
            model_name="anthropic/test-judge",
        )
        async def _judge() -> _AnthropicResponse:
            return _AnthropicResponse(input_tokens=40, output_tokens=60)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/faithfulness"] == pytest.approx(0.00062)

    @pytest.mark.asyncio
    async def test_decorator_fails_fast_without_model_aware_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        @track_api_cost(source="llm_judge/missing_pricing", provider="openai")
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=10, completion_tokens=20)

        token = builder.activate()
        try:
            with pytest.raises(ValueError, match="model-aware pricing"):
                await _judge()
        finally:
            token.var.reset(token)

    @pytest.mark.asyncio
    async def test_custom_extractor_takes_precedence(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.register_cost_extractor("openai", lambda _response: 0.75)

        @track_api_cost(
            source="llm_judge/custom",
            provider="openai",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=1, completion_tokens=1)

        token = builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/train/llm_judge/custom"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_decorator_noops_without_active_builder(self) -> None:
        @track_api_cost(source="llm_judge/no_context", provider="openai")
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=10, completion_tokens=20)

        result = await _judge()
        assert isinstance(result, _OpenAIResponse)

    @pytest.mark.asyncio
    async def test_for_cost_context_routes_to_eval_and_shares_state(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        eval_builder = builder.for_cost_context("eval")

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        token = eval_builder.activate()
        try:
            await _judge()
        finally:
            token.var.reset(token)

        metrics = await builder.flush()
        assert metrics["costs/eval/llm_judge/correctness"] == pytest.approx(0.0002)


class TestTrackApiCostIntegration:
    @pytest.mark.asyncio
    async def test_model_log_emits_train_and_eval_costs(self, tmp_path: Path) -> None:
        model = Model(
            name="metrics-cost-test",
            project="metrics-cost-test",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _train_judge() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        @track_api_cost(
            source="llm_judge/factuality",
            provider="anthropic",
            prompt_price_per_million=3.0,
            completion_price_per_million=4.0,
        )
        async def _eval_judge() -> _AnthropicResponse:
            return _AnthropicResponse(input_tokens=40, output_tokens=10)

        train_token = model.activate_metrics_context("train")
        try:
            await _train_judge()
        finally:
            train_token.var.reset(train_token)

        await model.log(trajectories=None, split="train", step=1, metrics={})

        eval_token = model.activate_metrics_context("eval")
        try:
            await _eval_judge()
        finally:
            eval_token.var.reset(eval_token)

        await model.log(trajectories=None, split="val", step=2, metrics={})

        history_path = (
            tmp_path
            / "metrics-cost-test"
            / "models"
            / "metrics-cost-test"
            / "history.jsonl"
        )
        with open(history_path) as f:
            first = json.loads(f.readline())
            second = json.loads(f.readline())

        assert first["costs/train/llm_judge/correctness"] == pytest.approx(0.0002)
        assert second["costs/eval/llm_judge/factuality"] == pytest.approx(0.00016)
        assert second["costs/cum/all"] == pytest.approx(0.00036)

    @pytest.mark.asyncio
    async def test_pipeline_trainer_activates_train_context_for_rollouts(
        self, tmp_path: Path
    ) -> None:
        model = TrainableModel(
            name="pipeline-context-test",
            project="pipeline-context-test",
            base_model="test-model",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        backend = MagicMock()
        observed_contexts: list[str] = []

        async def rollout_fn(
            _model: TrainableModel,
            _scenario: dict,
            _config: dict,
        ) -> TrajectoryGroup:
            observed_contexts.append(MetricsBuilder.get_active().cost_context)
            return TrajectoryGroup(
                [
                    Trajectory(
                        reward=1.0,
                        messages_and_choices=[
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "hi"},
                        ],
                    )
                ]
            )

        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=rollout_fn,
            scenarios=[{"metadata": {"scenario_id": "s1"}}],
            config={},
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
            eval_fn=None,
        )
        trainer._output_queue = asyncio.Queue()

        await trainer._rollout_worker(worker_id=0)

        assert observed_contexts == ["train"]

    @pytest.mark.asyncio
    async def test_pipeline_trainer_activates_eval_context_for_eval_fn(
        self, tmp_path: Path
    ) -> None:
        model = TrainableModel(
            name="pipeline-eval-context-test",
            project="pipeline-eval-context-test",
            base_model="test-model",
            base_path=str(tmp_path),
            report_metrics=[],
        )
        backend = MagicMock()
        observed_contexts: list[str] = []

        @track_api_cost(
            source="llm_judge/correctness",
            provider="openai",
            prompt_price_per_million=1.0,
            completion_price_per_million=2.0,
        )
        async def _judge_call() -> _OpenAIResponse:
            return _OpenAIResponse(prompt_tokens=100, completion_tokens=50)

        async def eval_fn(
            _model: TrainableModel,
            _step: int,
            _config: dict,
        ) -> list[Trajectory]:
            observed_contexts.append(MetricsBuilder.get_active().cost_context)
            await _judge_call()
            return [
                Trajectory(
                    reward=1.0,
                    messages_and_choices=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                )
            ]

        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
            scenarios=[],
            config={},
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
            eval_fn=eval_fn,
        )

        await trainer._run_eval(step=1)

        assert observed_contexts == ["eval"]

        history_path = (
            tmp_path
            / "pipeline-eval-context-test"
            / "models"
            / "pipeline-eval-context-test"
            / "history.jsonl"
        )
        with open(history_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert any("costs/eval/llm_judge/correctness" in row for row in rows)
        assert any("time/step_eval_s" in row for row in rows)
