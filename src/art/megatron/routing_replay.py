from __future__ import annotations

import json
from pathlib import Path
import re
import types
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, model_validator
from safetensors.torch import load_file, save_file
import torch

ROUTER_NAME_TOKEN = ".mlp.router"
ROUTER_KEY_FORMAT_VERSION = "moe_routing_replay_v1"
GLOBAL_TOKEN_UIDS_KEY = "global_token_uids"

_ROUTER_LAYER_PATTERN = re.compile(r"decoder\.layers\.(?P<layer>\d+)\.mlp\.router$")
_TRACE_CHUNK_PREFIX_PATTERN = re.compile(r"^chunk(?P<chunk>\d+)\.(?P<name>.+)$")


def _to_tensor_cpu_contiguous(
    tensor: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _normalize_step_index(step_index: int) -> str:
    if step_index < 0:
        raise ValueError(f"step_index must be non-negative, got {step_index}")
    return f"{step_index:06d}"


def _build_tensor_key(router_key: str, call_index: int, field_name: str) -> str:
    return f"{router_key}/call_{call_index}/{field_name}"


def _flatten_router_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise RuntimeError(
            f"Router tensor must have rank >=2, got shape={tuple(tensor.shape)}"
        )
    num_experts = int(tensor.shape[-1])
    return tensor.reshape(-1, num_experts).contiguous()


def _extract_router_output_tensors(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        probs, routing_map = output[0], output[1]
    elif isinstance(output, dict):
        probs = output.get("probs")
        routing_map = output.get("routing_map")
    else:
        raise RuntimeError(f"Unsupported router output type: {type(output)}")

    if not isinstance(probs, torch.Tensor):
        raise RuntimeError(f"Expected probs tensor, got {type(probs)}")
    if not isinstance(routing_map, torch.Tensor):
        raise RuntimeError(f"Expected routing_map tensor, got {type(routing_map)}")

    probs_2d = _flatten_router_tensor(probs.to(torch.float32))
    routing_map_2d = _flatten_router_tensor(routing_map.bool())
    if probs_2d.shape != routing_map_2d.shape:
        raise RuntimeError(
            "Router output shape mismatch: "
            f"probs={tuple(probs_2d.shape)} routing_map={tuple(routing_map_2d.shape)}"
        )
    return probs_2d, routing_map_2d


def build_router_key_from_module_name(*, chunk_index: int, module_name: str) -> str:
    match = _ROUTER_LAYER_PATTERN.search(module_name)
    if match is None:
        raise RuntimeError(
            f"Unable to derive router key from module name '{module_name}'. "
            f"Expected suffix matching '{_ROUTER_LAYER_PATTERN.pattern}'."
        )
    layer_index = int(match.group("layer"))
    return f"chunk_{chunk_index:02d}.layer_{layer_index:04d}.mlp.router"


def build_router_key_from_trace_name(trace_module_name: str) -> str:
    chunk_match = _TRACE_CHUNK_PREFIX_PATTERN.match(trace_module_name)
    if chunk_match is None:
        raise RuntimeError(
            "Forward trace router module name must start with 'chunk<idx>.'; "
            f"got '{trace_module_name}'"
        )
    chunk_index = int(chunk_match.group("chunk"))
    module_name = chunk_match.group("name")
    return build_router_key_from_module_name(
        chunk_index=chunk_index,
        module_name=module_name,
    )


class ParallelTopology(BaseModel):
    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: bool = False
    cp: int = 1
    pp: int = 1
    vpp: int = 1


class RouterCallRoute(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expert_indices: torch.Tensor
    expert_probs: torch.Tensor
    expert_mask: torch.Tensor
    routing_map: torch.Tensor | None = None
    num_experts: int

    @model_validator(mode="after")
    def _validate(self) -> "RouterCallRoute":
        self.expert_indices = _to_tensor_cpu_contiguous(
            self.expert_indices, dtype=torch.int32
        )
        self.expert_probs = _to_tensor_cpu_contiguous(
            self.expert_probs, dtype=torch.float32
        )
        self.expert_mask = _to_tensor_cpu_contiguous(self.expert_mask, dtype=torch.bool)
        if self.routing_map is not None:
            self.routing_map = _to_tensor_cpu_contiguous(
                self.routing_map, dtype=torch.bool
            )

        if self.expert_indices.ndim != 2:
            raise RuntimeError(
                "expert_indices must have shape [num_tokens, max_topk], got "
                f"{tuple(self.expert_indices.shape)}"
            )
        if self.expert_probs.shape != self.expert_indices.shape:
            raise RuntimeError(
                "expert_probs shape must match expert_indices shape, got "
                f"{tuple(self.expert_probs.shape)} vs {tuple(self.expert_indices.shape)}"
            )
        if self.expert_mask.shape != self.expert_indices.shape:
            raise RuntimeError(
                "expert_mask shape must match expert_indices shape, got "
                f"{tuple(self.expert_mask.shape)} vs {tuple(self.expert_indices.shape)}"
            )
        if self.num_experts <= 0:
            raise RuntimeError(f"num_experts must be >0, got {self.num_experts}")
        if self.routing_map is not None:
            expected = (self.expert_indices.shape[0], self.num_experts)
            if tuple(self.routing_map.shape) != expected:
                raise RuntimeError(
                    "routing_map shape mismatch: "
                    f"expected={expected}, got={tuple(self.routing_map.shape)}"
                )
        return self

    @property
    def num_global_tokens(self) -> int:
        return int(self.expert_indices.shape[0])

    @property
    def max_topk(self) -> int:
        return int(self.expert_indices.shape[1])


class StepRouterRoutes(BaseModel):
    calls: dict[int, RouterCallRoute]

    @model_validator(mode="after")
    def _validate_calls(self) -> "StepRouterRoutes":
        if not self.calls:
            raise RuntimeError("StepRouterRoutes.calls cannot be empty")
        for call_index in self.calls:
            if call_index < 0:
                raise RuntimeError(f"call_index must be >=0, got {call_index}")
        return self


class StepRoutes(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    routers: dict[str, StepRouterRoutes]
    global_token_uids: torch.Tensor

    @model_validator(mode="after")
    def _validate(self) -> "StepRoutes":
        if not self.routers:
            raise RuntimeError("StepRoutes.routers cannot be empty")
        self.global_token_uids = _to_tensor_cpu_contiguous(
            self.global_token_uids, dtype=torch.int64
        )
        if self.global_token_uids.ndim != 1:
            raise RuntimeError(
                "global_token_uids must have shape [num_global_tokens], got "
                f"{tuple(self.global_token_uids.shape)}"
            )
        if int(torch.unique(self.global_token_uids).numel()) != int(
            self.global_token_uids.numel()
        ):
            raise RuntimeError("global_token_uids must be unique per step")
        expected_tokens = int(self.global_token_uids.numel())
        for router_key, step_router in self.routers.items():
            for call_index, route in step_router.calls.items():
                if route.num_global_tokens != expected_tokens:
                    raise RuntimeError(
                        "Route token count mismatch for "
                        f"router='{router_key}' call={call_index}: "
                        f"route_tokens={route.num_global_tokens}, "
                        f"expected_tokens={expected_tokens}"
                    )
        return self


class MoeRoutingReplayBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    format_version: str = ROUTER_KEY_FORMAT_VERSION
    topology: ParallelTopology
    num_steps: int
    max_topk: int
    router_keys: list[str]
    steps: dict[int, StepRoutes]

    @model_validator(mode="after")
    def _validate(self) -> "MoeRoutingReplayBundle":
        if self.format_version != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported format_version={self.format_version}; "
                f"expected={ROUTER_KEY_FORMAT_VERSION}"
            )
        if self.num_steps <= 0:
            raise RuntimeError(f"num_steps must be >0, got {self.num_steps}")
        if self.max_topk < 0:
            raise RuntimeError(f"max_topk must be >=0, got {self.max_topk}")
        if set(self.steps.keys()) != set(range(self.num_steps)):
            raise RuntimeError(
                "steps must be indexed from 0..num_steps-1 without gaps: "
                f"num_steps={self.num_steps}, step_keys={sorted(self.steps.keys())}"
            )
        if not self.router_keys:
            raise RuntimeError("router_keys cannot be empty")
        router_key_set = set(self.router_keys)
        for step_index, step_routes in self.steps.items():
            step_router_keys = set(step_routes.routers.keys())
            if step_router_keys != router_key_set:
                raise RuntimeError(
                    f"Step {step_index} router set mismatch. "
                    f"expected={sorted(router_key_set)}, got={sorted(step_router_keys)}"
                )
        return self

    @classmethod
    def from_dir(cls, bundle_dir: str | Path) -> "MoeRoutingReplayBundle":
        base_dir = Path(bundle_dir)
        manifest_path = base_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing routing replay manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("format_version") != ROUTER_KEY_FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported routing replay manifest version: "
                f"{manifest.get('format_version')}"
            )

        topology = ParallelTopology.model_validate(manifest["topology"])
        num_steps = int(manifest["num_steps"])
        max_topk = int(manifest["max_topk"])
        router_keys = [str(key) for key in manifest["router_keys"]]
        manifest_steps = manifest["steps"]

        steps: dict[int, StepRoutes] = {}
        for step_index in range(num_steps):
            step_manifest = manifest_steps[str(step_index)]
            step_file = base_dir / step_manifest["file"]
            if not step_file.exists():
                raise FileNotFoundError(
                    f"Missing routing replay step file for step={step_index}: {step_file}"
                )
            step_tensors = load_file(str(step_file))
            if GLOBAL_TOKEN_UIDS_KEY not in step_tensors:
                raise RuntimeError(
                    f"Step file missing '{GLOBAL_TOKEN_UIDS_KEY}': {step_file}"
                )
            global_token_uids = step_tensors[GLOBAL_TOKEN_UIDS_KEY]

            routers: dict[str, StepRouterRoutes] = {}
            for router_key in router_keys:
                router_step_manifest = step_manifest["routers"].get(router_key)
                if router_step_manifest is None:
                    raise RuntimeError(
                        f"Step manifest missing router_key='{router_key}' for step={step_index}"
                    )
                calls: dict[int, RouterCallRoute] = {}
                for call_index_raw, call_manifest in router_step_manifest.items():
                    call_index = int(call_index_raw)
                    expert_indices_key = _build_tensor_key(
                        router_key, call_index, "expert_indices"
                    )
                    expert_probs_key = _build_tensor_key(
                        router_key, call_index, "expert_probs"
                    )
                    expert_mask_key = _build_tensor_key(
                        router_key, call_index, "expert_mask"
                    )
                    routing_map_key = _build_tensor_key(
                        router_key, call_index, "routing_map"
                    )
                    if expert_indices_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_indices_key}' in {step_file}"
                        )
                    if expert_probs_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_probs_key}' in {step_file}"
                        )
                    if expert_mask_key not in step_tensors:
                        raise RuntimeError(
                            f"Missing tensor key '{expert_mask_key}' in {step_file}"
                        )
                    routing_map = (
                        step_tensors[routing_map_key]
                        if routing_map_key in step_tensors
                        else None
                    )
                    calls[call_index] = RouterCallRoute(
                        expert_indices=step_tensors[expert_indices_key],
                        expert_probs=step_tensors[expert_probs_key],
                        expert_mask=step_tensors[expert_mask_key],
                        routing_map=routing_map,
                        num_experts=int(call_manifest["num_experts"]),
                    )
                routers[router_key] = StepRouterRoutes(calls=calls)
            steps[step_index] = StepRoutes(
                routers=routers,
                global_token_uids=global_token_uids,
            )

        return cls(
            format_version=ROUTER_KEY_FORMAT_VERSION,
            topology=topology,
            num_steps=num_steps,
            max_topk=max_topk,
            router_keys=router_keys,
            steps=steps,
        )

    def to_dir(self, bundle_dir: str | Path) -> None:
        base_dir = Path(bundle_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        manifest_steps: dict[str, dict[str, Any]] = {}
        for step_index in range(self.num_steps):
            step_routes = self.steps[step_index]
            step_file_name = f"step_{_normalize_step_index(step_index)}.safetensors"
            step_file_path = base_dir / step_file_name
            step_tensors: dict[str, torch.Tensor] = {
                GLOBAL_TOKEN_UIDS_KEY: _to_tensor_cpu_contiguous(
                    step_routes.global_token_uids, dtype=torch.int64
                )
            }
            step_manifest_routers: dict[str, dict[str, dict[str, int]]] = {}
            for router_key in self.router_keys:
                router_routes = step_routes.routers[router_key]
                call_manifest: dict[str, dict[str, int]] = {}
                for call_index, route in sorted(router_routes.calls.items()):
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_indices")
                    ] = _to_tensor_cpu_contiguous(
                        route.expert_indices, dtype=torch.int32
                    )
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_probs")
                    ] = _to_tensor_cpu_contiguous(
                        route.expert_probs, dtype=torch.float32
                    )
                    step_tensors[
                        _build_tensor_key(router_key, call_index, "expert_mask")
                    ] = _to_tensor_cpu_contiguous(route.expert_mask, dtype=torch.bool)
                    if route.routing_map is not None:
                        step_tensors[
                            _build_tensor_key(router_key, call_index, "routing_map")
                        ] = _to_tensor_cpu_contiguous(
                            route.routing_map, dtype=torch.bool
                        )
                    call_manifest[str(call_index)] = {"num_experts": route.num_experts}
                step_manifest_routers[router_key] = call_manifest
            save_file(step_tensors, str(step_file_path))
            manifest_steps[str(step_index)] = {
                "file": step_file_name,
                "routers": step_manifest_routers,
            }

        manifest = {
            "format_version": ROUTER_KEY_FORMAT_VERSION,
            "topology": self.topology.model_dump(mode="json"),
            "num_steps": self.num_steps,
            "max_topk": self.max_topk,
            "router_keys": self.router_keys,
            "steps": manifest_steps,
        }
        with (base_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)


class LocalTokenIndexer(Protocol):
    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        """Build local token uid order for current rank."""


class TopologyAwareLocalTokenIndexer:
    def __init__(self, parallel_state_module: Any | None = None) -> None:
        self._parallel_state = parallel_state_module

    def _ps(self) -> Any:
        if self._parallel_state is not None:
            return self._parallel_state
        from megatron.core import parallel_state as ps

        self._parallel_state = ps
        return ps

    def build_local_token_uids(
        self,
        *,
        global_token_uids: torch.Tensor,
        num_local_tokens: int,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> torch.Tensor:
        ps = self._ps()

        local_uids = global_token_uids.to(dtype=torch.int64, device="cpu").view(1, -1)

        cp_size = int(ps.get_context_parallel_world_size())
        if context_parallel_size > 1 and cp_size > 1:
            from megatron.core.utils import get_batch_on_this_cp_rank

            local_uids = get_batch_on_this_cp_rank({"tokens": local_uids})["tokens"]

        tp_size = int(ps.get_tensor_model_parallel_world_size())
        tp_rank = int(ps.get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        if sequence_parallel and tp_size > 1:
            tokens_per_tp_rank = local_uids.shape[1] // tp_size
            start = tp_rank * tokens_per_tp_rank
            local_uids = local_uids[:, start : start + tokens_per_tp_rank]

        return local_uids.reshape(-1).contiguous()


def _patch_alltoall_dispatcher_preprocess() -> None:
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )
    except Exception:
        return

    if hasattr(MoEAlltoAllTokenDispatcher, "_art_router_replay_preprocess_patched"):
        return

    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(
        self: Any, routing_map: torch.Tensor, *args: Any, **kwargs: Any
    ):
        result = original_preprocess(self, routing_map, *args, **kwargs)
        if (
            not getattr(self, "drop_and_pad", False)
            and getattr(self.config, "moe_expert_capacity_factor", None) is None
            and not (
                getattr(self.config, "moe_router_padding_for_quantization", None)
                or getattr(self.config, "moe_router_padding_for_fp8", None)
            )
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    setattr(MoEAlltoAllTokenDispatcher, "preprocess", patched_preprocess)
    setattr(MoEAlltoAllTokenDispatcher, "_art_router_replay_preprocess_patched", True)


class MoeRoutingReplayController:
    def __init__(
        self,
        *,
        bundle: MoeRoutingReplayBundle,
        strict: bool,
        local_token_indexer: LocalTokenIndexer | None = None,
    ) -> None:
        self.bundle = bundle
        self.strict = strict
        self.local_token_indexer = (
            local_token_indexer or TopologyAwareLocalTokenIndexer()
        )

        self._active_step_index: int | None = None
        self._active_sample_index: int | None = None
        self._active_step_routes: StepRoutes | None = None
        self._router_call_cursors: dict[str, int] = {}
        self._router_call_limits: dict[str, int] = {}
        self._global_uid_to_row_index: dict[int, int] = {}
        self._local_router_keys: set[str] = set()

        self._patched_router_modules: list[dict[str, Any]] = []

    def install_router_patches(self, model_chunks: list[Any]) -> None:
        if self._patched_router_modules:
            return
        _patch_alltoall_dispatcher_preprocess()

        for chunk_index, chunk in enumerate(model_chunks):
            for module_name, module in chunk.named_modules():
                if ROUTER_NAME_TOKEN not in module_name:
                    continue
                if not hasattr(module, "routing"):
                    continue
                router_key = build_router_key_from_module_name(
                    chunk_index=chunk_index,
                    module_name=module_name,
                )
                if self.strict and router_key not in self.bundle.router_keys:
                    raise RuntimeError(
                        "Router key from model is missing in replay bundle: "
                        f"router_key='{router_key}'"
                    )

                original_routing = module.routing
                if getattr(module, "_art_router_replay_patched", False):
                    continue

                sequence_parallel = bool(
                    getattr(getattr(module, "config", None), "sequence_parallel", False)
                )
                context_parallel_size = int(
                    getattr(getattr(module, "config", None), "context_parallel_size", 1)
                )

                def routing_wrapper(
                    _module: Any,
                    logits: torch.Tensor,
                    *args: Any,
                    _router_key: str = router_key,
                    _sequence_parallel: bool = sequence_parallel,
                    _context_parallel_size: int = context_parallel_size,
                    **kwargs: Any,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    live_probs, live_routing_map = original_routing(
                        logits, *args, **kwargs
                    )
                    replay_probs, replay_routing_map = self.get_route_for_router(
                        router_key=_router_key,
                        logits=live_probs,
                        sequence_parallel=_sequence_parallel,
                        context_parallel_size=_context_parallel_size,
                    )
                    # same result, but autograd goes through
                    probs = (
                        live_probs
                        + (
                            replay_probs.to(
                                device=live_probs.device,
                                dtype=live_probs.dtype,
                            )
                            - live_probs
                        ).detach()
                    )
                    routing_map = replay_routing_map.to(
                        device=live_routing_map.device,
                        dtype=live_routing_map.dtype,
                    )
                    return probs, routing_map

                module.routing = types.MethodType(routing_wrapper, module)
                module._art_router_replay_patched = True
                self._local_router_keys.add(router_key)
                self._patched_router_modules.append(
                    {
                        "module": module,
                        "router_key": router_key,
                        "original_routing": original_routing,
                    }
                )

    def remove_router_patches(self) -> None:
        for item in self._patched_router_modules:
            module = item["module"]
            module.routing = item["original_routing"]
            if hasattr(module, "_art_router_replay_patched"):
                delattr(module, "_art_router_replay_patched")
        self._patched_router_modules.clear()
        self._local_router_keys.clear()

    def set_step(self, *, step_index: int, sample_index: int) -> None:
        from megatron.core import parallel_state as ps

        if step_index not in self.bundle.steps:
            raise RuntimeError(
                f"Replay bundle missing step_index={step_index}. "
                f"Available steps={sorted(self.bundle.steps.keys())}"
            )
        step_routes = self.bundle.steps[step_index]
        self._active_step_index = step_index
        self._active_sample_index = sample_index
        self._active_step_routes = step_routes
        for local_router_key in sorted(self._local_router_keys):
            if local_router_key not in step_routes.routers:
                raise RuntimeError(
                    "Replay bundle step is missing local router key: "
                    f"step={step_index}, router='{local_router_key}'"
                )
        dp_world_size = int(ps.get_data_parallel_world_size(with_context_parallel=True))
        dp_rank = int(ps.get_data_parallel_rank(with_context_parallel=True))
        self._router_call_cursors = {}
        self._router_call_limits = {}
        for router_key in sorted(self._local_router_keys):
            total_calls = len(step_routes.routers[router_key].calls)
            call_start = 0
            call_limit = total_calls
            if dp_world_size > 1:
                if total_calls % dp_world_size != 0:
                    raise RuntimeError(
                        "Replay router call count is not divisible by DP world size: "
                        f"step={step_index}, router='{router_key}', "
                        f"calls={total_calls}, dp_world_size={dp_world_size}"
                    )
                calls_per_dp_rank = total_calls // dp_world_size
                call_start = dp_rank * calls_per_dp_rank
                call_limit = call_start + calls_per_dp_rank
            self._router_call_cursors[router_key] = call_start
            self._router_call_limits[router_key] = call_limit
        self._global_uid_to_row_index = {
            int(uid.item()): row_index
            for row_index, uid in enumerate(step_routes.global_token_uids)
        }

    def finalize_step(self) -> None:
        if self._active_step_routes is None:
            raise RuntimeError("finalize_step called before set_step")
        for router_key in sorted(self._local_router_keys):
            consumed = self._router_call_cursors.get(router_key, 0)
            expected = self._router_call_limits.get(router_key)
            if expected is None:
                raise RuntimeError(
                    "Routing replay call limits missing for router key: "
                    f"step={self._active_step_index}, router='{router_key}'"
                )
            if consumed != expected:
                raise RuntimeError(
                    "Routing replay step consumption mismatch: "
                    f"step={self._active_step_index}, router='{router_key}', "
                    f"consumed={consumed}, expected={expected}"
                )
        self._active_step_index = None
        self._active_sample_index = None
        self._active_step_routes = None
        self._router_call_cursors = {}
        self._router_call_limits = {}
        self._global_uid_to_row_index = {}

    def get_route_for_router(
        self,
        *,
        router_key: str,
        logits: torch.Tensor,
        sequence_parallel: bool,
        context_parallel_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_routes = self._active_step_routes
        call_index = self._router_call_cursors.get(router_key, 0)
        call_limit = self._router_call_limits.get(router_key)
        router_calls = step_routes.routers[router_key].calls
        if call_limit is not None and call_index >= call_limit:
            raise RuntimeError(
                "Routing replay call cursor exceeded local call range: "
                f"step={self._active_step_index}, router='{router_key}', "
                f"call_index={call_index}, limit={call_limit}"
            )
        route = router_calls[call_index]
        self._router_call_cursors[router_key] = call_index + 1

        num_local_tokens = int(logits.shape[0])
        num_experts = int(logits.shape[1])

        local_uids = self.local_token_indexer.build_local_token_uids(
            global_token_uids=step_routes.global_token_uids,
            num_local_tokens=num_local_tokens,
            sequence_parallel=sequence_parallel,
            context_parallel_size=context_parallel_size,
        )
        row_index_tensor = torch.tensor(
            [self._global_uid_to_row_index[int(uid)] for uid in local_uids.tolist()],
            dtype=torch.int64,
        )

        local_indices = route.expert_indices.index_select(0, row_index_tensor)
        local_probs = route.expert_probs.index_select(0, row_index_tensor)
        local_mask = route.expert_mask.index_select(0, row_index_tensor)

        probs = torch.zeros(
            (num_local_tokens, num_experts),
            dtype=logits.dtype,
            device=logits.device,
        )
        routing_map = torch.zeros(
            (num_local_tokens, num_experts),
            dtype=torch.bool,
            device=logits.device,
        )

        if local_indices.numel() > 0:
            indices_device = local_indices.to(device=logits.device, dtype=torch.long)
            probs_device = local_probs.to(device=logits.device, dtype=logits.dtype)
            mask_device = local_mask.to(device=logits.device, dtype=torch.bool)
            row_index_device = (
                torch.arange(num_local_tokens, device=logits.device)
                .unsqueeze(1)
                .expand_as(indices_device)
            )

            selected_rows = row_index_device[mask_device]
            selected_cols = indices_device[mask_device]
            selected_probs = probs_device[mask_device]

            if selected_rows.numel() > 0:
                probs[selected_rows, selected_cols] = selected_probs
                routing_map[selected_rows, selected_cols] = True

        return probs, routing_map


def _compact_route_from_dense(
    probs_2d: torch.Tensor,
    routing_map_2d: torch.Tensor,
) -> RouterCallRoute:
    num_tokens, num_experts = probs_2d.shape
    if num_tokens == 0:
        return RouterCallRoute(
            expert_indices=torch.zeros((0, 0), dtype=torch.int32),
            expert_probs=torch.zeros((0, 0), dtype=torch.float32),
            expert_mask=torch.zeros((0, 0), dtype=torch.bool),
            num_experts=num_experts,
        )

    max_topk = int(routing_map_2d.sum(dim=1).max().item())
    expert_indices = torch.zeros((num_tokens, max_topk), dtype=torch.int32)
    expert_probs = torch.zeros((num_tokens, max_topk), dtype=torch.float32)
    expert_mask = torch.zeros((num_tokens, max_topk), dtype=torch.bool)
    for token_index in range(num_tokens):
        expert_ids = torch.nonzero(
            routing_map_2d[token_index], as_tuple=False
        ).flatten()
        slot_count = int(expert_ids.numel())
        if slot_count == 0:
            continue
        expert_indices[token_index, :slot_count] = expert_ids.to(torch.int32)
        expert_probs[token_index, :slot_count] = probs_2d[token_index, expert_ids].to(
            torch.float32
        )
        expert_mask[token_index, :slot_count] = True

    return RouterCallRoute(
        expert_indices=expert_indices,
        expert_probs=expert_probs,
        expert_mask=expert_mask,
        num_experts=num_experts,
    )


def build_bundle_from_forward_trace_dir(
    *,
    traces_dir: str | Path,
    num_steps: int,
    topology: ParallelTopology,
) -> MoeRoutingReplayBundle:
    """Build a replay bundle from saved forward traces for the correctness harness.

    This helper is intended for testing/oracle routing replay workflows and is not
    part of inference routing capture/export.
    """
    trace_dir = Path(traces_dir)
    steps: dict[int, StepRoutes] = {}
    router_keys_union: set[str] = set()
    max_topk = 0

    for step_index in range(num_steps):
        trace_path = trace_dir / f"forward_trace_step_{step_index:03d}.pt"
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Missing forward trace for step={step_index}: {trace_path}"
            )
        step_trace: dict[str, list[dict[str, Any]]] = torch.load(
            trace_path, map_location="cpu", weights_only=False
        )

        step_routers: dict[str, StepRouterRoutes] = {}
        step_global_tokens: int | None = None
        for module_name in sorted(step_trace.keys()):
            if ROUTER_NAME_TOKEN not in module_name:
                continue
            router_key = build_router_key_from_trace_name(module_name)
            router_calls: dict[int, RouterCallRoute] = {}
            for call_index, call_entry in enumerate(step_trace[module_name]):
                output = call_entry.get("output")
                probs_2d, routing_map_2d = _extract_router_output_tensors(output)
                compact_route = _compact_route_from_dense(probs_2d, routing_map_2d)
                router_calls[call_index] = compact_route
                max_topk = max(max_topk, compact_route.max_topk)
                token_count = compact_route.num_global_tokens
                if step_global_tokens is None:
                    step_global_tokens = token_count
                elif step_global_tokens != token_count:
                    raise RuntimeError(
                        "Inconsistent token count across routers within step: "
                        f"step={step_index}, expected={step_global_tokens}, got={token_count}, "
                        f"router='{router_key}', call={call_index}"
                    )

            if not router_calls:
                raise RuntimeError(
                    f"Router trace has no calls for module '{module_name}' at step={step_index}"
                )
            step_routers[router_key] = StepRouterRoutes(calls=router_calls)
            router_keys_union.add(router_key)

        if not step_routers:
            raise RuntimeError(
                f"No router traces found for step={step_index} in {trace_path}"
            )
        if step_global_tokens is None:
            raise RuntimeError(
                f"Could not infer token count for step={step_index} from router traces"
            )
        global_token_uids = torch.arange(step_global_tokens, dtype=torch.int64)
        steps[step_index] = StepRoutes(
            routers=step_routers,
            global_token_uids=global_token_uids,
        )

    router_keys = sorted(router_keys_union)
    for step_index, step_routes in steps.items():
        if set(step_routes.routers.keys()) != set(router_keys):
            raise RuntimeError(
                f"Step {step_index} router keys differ from global set: "
                f"step_keys={sorted(step_routes.routers.keys())}, router_keys={router_keys}"
            )

    return MoeRoutingReplayBundle(
        format_version=ROUTER_KEY_FORMAT_VERSION,
        topology=topology,
        num_steps=num_steps,
        max_topk=max_topk,
        router_keys=router_keys,
        steps=steps,
    )
