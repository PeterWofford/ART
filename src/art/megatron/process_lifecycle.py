from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from pathlib import Path

import torch


def _needs_setup() -> bool:
    try:
        import megatron.bridge  # type: ignore

        return False
    except ImportError:
        return True


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


async def ensure_megatron_running(
    process: asyncio.subprocess.Process | None,
    *,
    base_model: str,
    setup_script: Path,
    train_script: Path,
) -> asyncio.subprocess.Process:
    if process is not None and process.returncode is None:
        return process

    setup_cmd = f"bash {setup_script} && " if _needs_setup() else ""
    subprocess.run(["pkill", "-9", "megatron-service"], check=False)
    env = os.environ.copy()
    env["MODEL_IDENTIFIER"] = base_model
    megatron_visible = env.get("ART_MEGATRON_CUDA_VISIBLE_DEVICES")
    if megatron_visible:
        env["CUDA_VISIBLE_DEVICES"] = megatron_visible
        num_gpus = len([part for part in megatron_visible.split(",") if part.strip()])
    else:
        num_gpus = torch.cuda.device_count()
    master_port = env.get("MASTER_PORT") or str(_find_free_port())
    env["MASTER_PORT"] = master_port
    command = (
        f"{setup_cmd}uv run torchrun --nproc_per_node {num_gpus} "
        f"--master_port {master_port} {train_script}"
    )
    return await asyncio.create_subprocess_shell(command, env=env)


async def terminate_megatron_process(
    process: asyncio.subprocess.Process | None, *, timeout_s: float = 10.0
) -> None:
    if process is None:
        return
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
