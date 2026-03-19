from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.serverless.backend import ServerlessBackend


def _make_model() -> TrainableModel:
    model = TrainableModel(
        name="serverless-pipeline-compat",
        project="serverless-pipeline-compat",
        base_model="test-model",
        report_metrics=[],
    )
    model.id = "model-id"
    model.entity = "test-entity"
    return model


def _make_group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=1.0,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]
    )


@pytest.mark.asyncio
async def test_serverless_backend_accepts_pipeline_trainer_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with patch("art.serverless.backend.Client"):
        backend = ServerlessBackend(api_key="test-key")

    captured: dict[str, Any] = {}

    async def fake_train_model(
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: Any,
        dev_config: dict[str, Any],
        verbose: bool = False,
    ):
        captured["model"] = model
        captured["trajectory_groups"] = trajectory_groups
        captured["config"] = config
        captured["dev_config"] = dev_config
        captured["verbose"] = verbose
        yield {"loss/train": 1.5}

    async def fake_get_step(model: TrainableModel) -> int:
        assert model.id == "model-id"
        return 7

    monkeypatch.setattr(backend, "_train_model", fake_train_model)
    monkeypatch.setattr(backend, "_get_step", fake_get_step)

    model = _make_model()
    groups = [_make_group()]

    with pytest.warns(UserWarning, match="save_checkpoint=False"):
        result = await backend.train(
            model,
            groups,
            learning_rate=1e-5,
            loss_fn="ppo",
            save_checkpoint=False,
        )

    assert result.step == 7
    assert captured["model"] is model
    assert captured["trajectory_groups"] == groups
    assert captured["config"].learning_rate == 1e-5
    assert captured["dev_config"]["ppo"] is True
    assert result.metrics["loss/train"] == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_fn": "cispo", "ppo": True}, "loss_fn and ppo"),
        ({"loss_fn_config": {"foo": "bar"}}, "loss_fn_config=None"),
        ({"normalize_advantages": False}, "normalize_advantages=True"),
        ({"adam_params": object()}, "adam_params=None"),
    ],
)
async def test_serverless_backend_rejects_unsupported_pipeline_trainer_options(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with patch("art.serverless.backend.Client"):
        backend = ServerlessBackend(api_key="test-key")

    model = _make_model()
    groups = [_make_group()]

    with pytest.raises(ValueError, match=message):
        await backend.train(model, groups, **kwargs)
