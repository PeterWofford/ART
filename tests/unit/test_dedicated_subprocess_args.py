"""Unit tests for dedicated vLLM subprocess argument handling."""

import json
import subprocess

import httpx
import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("vllm")

from art.dev.model import InternalModelConfig
from art.unsloth.service import UnslothService


class _FakeProcess:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


class _FakeResponse:
    status_code = 200


class _FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):
        return _FakeResponse()


def _make_service(tmp_path):
    return UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-14B",
        config=InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[1, 2]),
        output_dir=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_start_subprocess_defaults_to_dp_for_multi_gpu(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    service = _make_service(tmp_path)
    await service._start_vllm_subprocess("/tmp/lora", 8000, config={})  # noqa: SLF001
    service.close()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    engine_args_json = next(
        arg.split("=", 1)[1] for arg in cmd if arg.startswith("--engine-args-json=")
    )
    engine_args = json.loads(engine_args_json)
    assert engine_args["data_parallel_size"] == 2
    assert engine_args["data_parallel_size_local"] == 2
    assert engine_args["distributed_executor_backend"] == "mp"


@pytest.mark.asyncio
async def test_start_subprocess_rejects_api_server_count_gt_one(tmp_path):
    service = _make_service(tmp_path)
    with pytest.raises(ValueError, match="api_server_count must be 1"):
        await service._start_vllm_subprocess(  # noqa: SLF001
            "/tmp/lora", 8000, config={"server_args": {"api_server_count": 2}}
        )


@pytest.mark.asyncio
async def test_start_subprocess_tp_rejects_dp_conflict(tmp_path):
    service = _make_service(tmp_path)
    with pytest.raises(ValueError, match="must be 1 or unset when tensor_parallel_size > 1"):
        await service._start_vllm_subprocess(  # noqa: SLF001
            "/tmp/lora",
            8000,
            config={
                "engine_args": {
                    "tensor_parallel_size": 2,
                    "data_parallel_size": 2,
                }
            },
        )


@pytest.mark.asyncio
async def test_start_subprocess_tp_does_not_auto_inject_dp_defaults(
    tmp_path, monkeypatch
):
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    service = _make_service(tmp_path)
    await service._start_vllm_subprocess(  # noqa: SLF001
        "/tmp/lora", 8000, config={"engine_args": {"tensor_parallel_size": 2}}
    )
    service.close()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    engine_args_json = next(
        arg.split("=", 1)[1] for arg in cmd if arg.startswith("--engine-args-json=")
    )
    engine_args = json.loads(engine_args_json)
    assert engine_args["tensor_parallel_size"] == 2
    assert "data_parallel_size" not in engine_args
    assert "data_parallel_size_local" not in engine_args
