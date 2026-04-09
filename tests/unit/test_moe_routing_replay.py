from __future__ import annotations

from pathlib import Path
import tempfile
from typing import cast

import pytest
import torch
from torch import nn

from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
    ParallelTopology,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
)


def _dense_from_compact(
    route: RouterCallRoute,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = route.expert_indices.shape[0]
    num_experts = route.num_experts
    probs = torch.zeros((num_tokens, num_experts), dtype=dtype)
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    for token_idx in range(num_tokens):
        for slot in range(route.expert_indices.shape[1]):
            if not bool(route.expert_mask[token_idx, slot]):
                continue
            expert_idx = int(route.expert_indices[token_idx, slot].item())
            probs[token_idx, expert_idx] = route.expert_probs[token_idx, slot].to(dtype)
            routing_map[token_idx, expert_idx] = True
    return probs, routing_map


def _make_bundle() -> tuple[MoeRoutingReplayBundle, RouterCallRoute]:
    router_key = "chunk_00.layer_0000.mlp.router"
    route = RouterCallRoute(
        expert_indices=torch.tensor(
            [
                [0, 2],
                [1, 0],
                [2, 1],
                [1, 0],
            ],
            dtype=torch.int32,
        ),
        expert_probs=torch.tensor(
            [
                [0.70, 0.30],
                [1.00, 0.00],
                [0.65, 0.35],
                [1.00, 0.00],
            ],
            dtype=torch.float32,
        ),
        expert_mask=torch.tensor(
            [
                [True, True],
                [True, False],
                [True, True],
                [True, False],
            ],
            dtype=torch.bool,
        ),
        num_experts=3,
    )
    bundle = MoeRoutingReplayBundle(
        topology=ParallelTopology(tp=1, ep=1, etp=1, dp=1, sp=False, cp=1, pp=1, vpp=1),
        num_steps=1,
        max_topk=2,
        router_keys=[router_key],
        steps={
            0: StepRoutes(
                routers={router_key: StepRouterRoutes(calls={0: route})},
                global_token_uids=torch.arange(4, dtype=torch.int64),
            )
        },
    )
    return bundle, route


class _IdentityIndexer:
    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        del sequence_parallel, context_parallel_size
        if int(global_token_uids.numel()) < num_local_tokens:
            raise RuntimeError("num_local_tokens exceeds global token count")
        return global_token_uids[:num_local_tokens].clone()


class _FakeRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"sequence_parallel": False, "context_parallel_size": 1},
        )()

    def routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1)
        routing_map = torch.zeros_like(logits, dtype=torch.bool)
        return probs, routing_map


class _FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _FakeRouter()


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeMlp()


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer()])


class _FakeChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _FakeDecoder()


def test_bundle_roundtrip_disk() -> None:
    bundle, route = _make_bundle()
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir)
        bundle.to_dir(bundle_path)
        loaded = MoeRoutingReplayBundle.from_dir(bundle_path)

    assert loaded.num_steps == 1
    assert loaded.max_topk == 2
    assert loaded.router_keys == bundle.router_keys
    loaded_route = loaded.steps[0].routers[bundle.router_keys[0]].calls[0]
    assert torch.equal(loaded_route.expert_indices, route.expert_indices)
    assert torch.equal(loaded_route.expert_probs, route.expert_probs)
    assert torch.equal(loaded_route.expert_mask, route.expert_mask)


def test_controller_patches_router_and_replays() -> None:
    bundle, route = _make_bundle()
    controller = MoeRoutingReplayController(
        bundle=bundle,
        strict=True,
        local_token_indexer=_IdentityIndexer(),
    )
    chunk = _FakeChunk()
    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=0)

    logits = torch.randn((4, 3), dtype=torch.float32)
    router = cast(
        _FakeRouter,
        chunk.decoder.layers[0].mlp.router,  # ty: ignore[possibly-missing-attribute]
    )
    replay_probs, replay_map = router.routing(logits)
    expected_probs, expected_map = _dense_from_compact(route, dtype=logits.dtype)

    assert torch.equal(replay_map.cpu(), expected_map)
    assert torch.allclose(replay_probs.cpu(), expected_probs, atol=0.0, rtol=0.0)

    controller.finalize_step()
    controller.remove_router_patches()


def test_controller_finalize_fails_when_unconsumed_calls_remain() -> None:
    bundle, _route = _make_bundle()
    controller = MoeRoutingReplayController(
        bundle=bundle,
        strict=True,
        local_token_indexer=_IdentityIndexer(),
    )
    chunk = _FakeChunk()
    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=0)
    with pytest.raises(RuntimeError, match="consumption mismatch"):
        controller.finalize_step()
