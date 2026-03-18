import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.dev.model import InternalModelConfig
from art.local import LocalBackend
from art.pipeline_trainer.trainer import PipelineTrainer
from art.utils.output_dirs import get_model_dir


def _make_group(
    rewards: list[float], *, initial_policy_version: int | None = 0
) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=initial_policy_version,
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{idx}"},
                    {"role": "assistant", "content": f"answer-{idx}"},
                ],
            )
            for idx, reward in enumerate(rewards)
        ]
    )


def _make_trainer(
    *,
    model: TrainableModel,
    backend: object,
    tmp_path: Path,
    **kwargs: Any,
) -> PipelineTrainer:
    return PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_batch_size=1,
        max_steps=1,
        eval_fn=None,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pipeline_trainer_preserves_default_backend_train_kwargs(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-default-backend-kwargs",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))
    loss_fn_config = {"alpha": 0.1}
    adam_params = object()

    trainer = _make_trainer(
        model=model,
        backend=backend,
        tmp_path=tmp_path,
        learning_rate=2e-5,
        loss_fn="cispo",
        loss_fn_config=loss_fn_config,
        normalize_advantages=True,
        adam_params=adam_params,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_count == 1
    assert backend.train.await_args.kwargs == {
        "learning_rate": 2e-5,
        "loss_fn": "cispo",
        "loss_fn_config": loss_fn_config,
        "normalize_advantages": True,
        "save_checkpoint": False,
        "adam_params": adam_params,
    }


@pytest.mark.asyncio
async def test_pipeline_trainer_translates_local_backend_kwargs(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-kwargs",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))  # type: ignore[method-assign]

    trainer = _make_trainer(
        model=model,
        backend=backend,
        tmp_path=tmp_path,
        learning_rate=3e-5,
        loss_fn="ppo",
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_count == 1  # type: ignore[attr-defined]
    assert backend.train.await_args.kwargs == {  # type: ignore[attr-defined]
        "learning_rate": 3e-5,
        "ppo": True,
        "save_checkpoint": False,
    }


@pytest.mark.asyncio
async def test_local_backend_close_awaits_async_service_cleanup(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    calls: list[str] = []

    class FakeService:
        async def aclose(self) -> None:
            calls.append("aclose")

    service = FakeService()
    backend._services["test-service"] = service  # type: ignore[assignment]

    with patch("art.local.backend.close_proxy") as close_proxy:
        await backend.close()

    assert calls == ["aclose"]
    close_proxy.assert_called_once_with(service)


@pytest.mark.asyncio
async def test_local_backend_async_context_manager_awaits_async_service_cleanup(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    calls: list[str] = []

    class FakeService:
        async def aclose(self) -> None:
            calls.append("aclose")

    service = FakeService()
    backend._services["test-service"] = service  # type: ignore[assignment]

    with patch("art.local.backend.close_proxy") as close_proxy:
        async with backend:
            pass

    assert calls == ["aclose"]
    close_proxy.assert_called_once_with(service)


@pytest.mark.asyncio
async def test_local_backend_sync_context_manager_warns_in_running_loop(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    calls: list[str] = []

    class FakeService:
        def close(self) -> None:
            calls.append("close")

    service = FakeService()
    backend._services["test-service"] = service  # type: ignore[assignment]

    with patch("art.local.backend.close_proxy") as close_proxy:
        with pytest.warns(RuntimeWarning, match="async with LocalBackend"):
            with backend:
                pass

    assert calls == ["close"]
    close_proxy.assert_called_once_with(service)


def test_local_backend_sync_context_manager_warns_and_uses_sync_close(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    calls: list[str] = []

    class FakeService:
        def close(self) -> None:
            calls.append("close")

        async def aclose(self) -> None:
            calls.append("aclose")

    service = FakeService()
    backend._services["test-service"] = service  # type: ignore[assignment]

    with patch("art.local.backend.close_proxy") as close_proxy:
        with pytest.warns(RuntimeWarning, match="best-effort sync shutdown"):
            with backend:
                pass

    assert calls == ["close"]
    close_proxy.assert_called_once_with(service)


@pytest.mark.parametrize(
    ("trainer_kwargs", "match"),
    [
        ({"loss_fn": "dro"}, "loss_fn='cispo' or loss_fn='ppo'"),
        ({"loss_fn_config": {"clip": 0.2}}, "loss_fn_config=None"),
        ({"normalize_advantages": False}, "normalize_advantages=True"),
        ({"adam_params": object()}, "adam_params=None"),
    ],
)
def test_pipeline_trainer_rejects_unsupported_local_backend_settings(
    tmp_path: Path,
    trainer_kwargs: dict[str, object],
    match: str,
) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-invalid",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    with pytest.raises(ValueError, match=match):
        _make_trainer(
            model=model,
            backend=backend,
            tmp_path=tmp_path,
            **trainer_kwargs,
        )


def test_pipeline_trainer_rejects_shared_local_backend(tmp_path: Path) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-shared",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = LocalBackend(path=str(tmp_path))

    with pytest.raises(
        ValueError, match="only supports LocalBackend in dedicated mode"
    ):
        _make_trainer(model=model, backend=backend, tmp_path=tmp_path)


def test_local_backend_inference_name_prefers_served_step_in_dedicated_mode(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="local-backend-served-step",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    output_dir = Path(get_model_dir(model=model, art_path=str(tmp_path)))
    (output_dir / "checkpoints" / "3").mkdir(parents=True)
    backend._services[model.name] = cast(Any, SimpleNamespace(_latest_step=2))

    assert backend._model_inference_name(model) == f"{model.name}@2"
    assert backend._model_inference_name(model, step=3) == f"{model.name}@3"
