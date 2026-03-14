from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

CAPTURE_NAME_TOKENS = (
    ".self_attention.linear_qkv",
    ".self_attention.linear_qkv.q_proj_lora",
    ".self_attention.linear_qkv.k_proj_lora",
    ".self_attention.linear_qkv.v_proj_lora",
    ".self_attention.linear_proj",
    ".self_attention.linear_proj.lora",
    ".mlp.router",
    ".mlp.experts.linear_fc1",
    ".mlp.experts.linear_fc1.gate_lora",
    ".mlp.experts.linear_fc1.up_lora",
    ".mlp.experts.linear_fc2",
    ".mlp.experts.linear_fc2.lora",
)
ROUTER_NAME_TOKEN = ".mlp.router"
PRIMARY_OUTPUT_CANONICAL_KEY = "primary_output__is_canonical"


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerces scalar values to int for trace metadata."""
    try:
        return int(value)
    except Exception:
        return default


def _safe_ps_stat(name: str, default: int) -> int:
    """Reads one Megatron parallel-state integer when available."""
    try:
        from megatron.core import parallel_state as ps

        getter = getattr(ps, name)
        return _safe_int(getter(), default)
    except Exception:
        return default


def _rank_metadata() -> dict[str, int]:
    """Builds lightweight distributed metadata for one trace call."""
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():
        rank = _safe_int(torch.distributed.get_rank(), 0)
        world_size = _safe_int(torch.distributed.get_world_size(), 1)
    return {
        "global_rank": rank,
        "world_size": world_size,
        "tp_rank": _safe_ps_stat("get_tensor_model_parallel_rank", 0),
        "tp_world_size": _safe_ps_stat("get_tensor_model_parallel_world_size", 1),
        "ep_rank": _safe_ps_stat("get_expert_model_parallel_rank", 0),
        "ep_world_size": _safe_ps_stat("get_expert_model_parallel_world_size", 1),
        "etp_rank": _safe_ps_stat("get_expert_tensor_parallel_rank", 0),
        "etp_world_size": _safe_ps_stat("get_expert_tensor_parallel_world_size", 1),
        "dp_rank": _safe_ps_stat("get_data_parallel_rank", 0),
        "dp_world_size": _safe_ps_stat("get_data_parallel_world_size", 1),
        "expert_dp_rank": _safe_ps_stat("get_expert_data_parallel_rank", 0),
        "expert_dp_world_size": _safe_ps_stat("get_expert_data_parallel_world_size", 1),
    }


def _shard_world_size_for_domain(domain: Any) -> int:
    """Returns shard-group world size for one LoRA shard domain."""
    if domain == "tp":
        return _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
    if domain == "expert_tp":
        return _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
    return 1


def _extract_primary_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = cast(torch.Tensor, full_tensor())
    else:
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            tensor = cast(torch.Tensor, to_local())
        else:
            local_tensor = getattr(tensor, "_local_tensor", None)
            if isinstance(local_tensor, torch.Tensor):
                tensor = local_tensor
    return tensor.detach().cpu()


def _materialize_trace_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _materialize_tensor(value)
    if isinstance(value, dict):
        return {key: _materialize_trace_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_trace_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_trace_value(item) for item in value)
    return value


def _extract_router_topk(output: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(output, tuple) or len(output) < 2:
        return None
    probs = output[0]
    routing_map = output[1]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return None
    probs = _materialize_tensor(probs.float())
    routing_map = _materialize_tensor(routing_map)
    topk = int(routing_map.sum(dim=-1).max().item())
    if topk < 0:
        raise RuntimeError(f"Invalid router topk={topk}")
    if topk == 0:
        topk_scores = probs.new_zeros((probs.shape[0], 0))
        topk_ids = torch.zeros((probs.shape[0], 0), dtype=torch.int64)
    else:
        topk_scores, topk_ids = torch.topk(probs, k=topk, dim=-1)
    return topk_ids.contiguous(), topk_scores.contiguous()


class ForwardTraceCapture:
    def __init__(
        self,
        model_chunks: list[Any],
        *,
        enabled: bool,
        capture_name_tokens: tuple[str, ...] = CAPTURE_NAME_TOKENS,
    ) -> None:
        self.enabled = enabled
        self.capture_name_tokens = capture_name_tokens
        self.current_step_index: int | None = None
        self.current_step_trace: dict[str, list[dict[str, Any]]] = {}
        self._hook_handles: list[Any] = []
        if not enabled:
            return
        self._register_hooks(model_chunks)

    def _register_hooks(self, model_chunks: list[Any]) -> None:
        for chunk_index, chunk in enumerate(model_chunks):
            for module_name, module in chunk.named_modules():
                trace_module_name = f"chunk{chunk_index}.{module_name}"
                is_layer_output = (
                    ".decoder.layers." in module_name
                    and module_name.rsplit(".", 1)[-1].isdigit()
                )
                if not is_layer_output and not any(
                    module_name.endswith(token) for token in self.capture_name_tokens
                ):
                    continue
                self._hook_handles.append(
                    module.register_forward_hook(
                        self._make_hook(trace_module_name, module)
                    )
                )

    @staticmethod
    def _sequence_parallel_enabled(module: Any) -> bool:
        """Returns sequence-parallel flag from module/provider/config when present."""
        for owner in (
            module,
            getattr(module, "provider", None),
            getattr(module, "config", None),
        ):
            if owner is None:
                continue
            value = getattr(owner, "sequence_parallel", None)
            if isinstance(value, bool):
                return value
        return False

    @staticmethod
    def _lora_primary_output_merge_hint(module: Any) -> dict[str, Any] | None:
        """Infers the correct output merge op for LoRA modules."""
        if module.__class__.__name__ != "LoRA":
            return None
        lora_module = module
        b_param = getattr(lora_module, "B_T", None)
        if b_param is None:
            return None
        b_domain = getattr(b_param, "lora_shard_domain", None)
        b_world_size = _shard_world_size_for_domain(b_domain)
        if bool(getattr(b_param, "lora_tp_sharded", False)) and b_world_size > 1:
            shard_dim = getattr(b_param, "lora_tp_shard_dim", None)
            if isinstance(shard_dim, int):
                return {"op": "concat", "dim": shard_dim}
        a_param = getattr(lora_module, "A_T", None)
        if a_param is None:
            return None
        a_domain = getattr(a_param, "lora_shard_domain", None)
        a_world_size = _shard_world_size_for_domain(a_domain)
        if bool(getattr(a_param, "lora_tp_sharded", False)) and a_world_size > 1:
            return {"op": "sum"}
        return None

    def _infer_primary_output_merge_hint(
        self, name: str, module: Any
    ) -> dict[str, Any] | None:
        """Chooses canonical cross-rank concat axis for one module output."""
        if ROUTER_NAME_TOKEN in name:
            return {"op": "concat", "dim": 0}

        lora_hint = self._lora_primary_output_merge_hint(module)
        if lora_hint is not None:
            return lora_hint

        # Base MoE expert linears need expert-TP aware merge semantics.
        # With etp>1:
        # - FC1 (column-parallel) shards output features -> concat on feature dim.
        # - FC2 (row-parallel) emits partial output contributions -> sum across ranks.
        # With etp==1, keep the existing token-row concat behavior.
        etp_world_size = _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
        if ".mlp.experts.linear_fc1" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {
                    "op": "concat",
                    "dim": -1,
                    "layout": "gate_up_rank_interleaved",
                }
            return {"op": "concat", "dim": 0}
        if ".mlp.experts.linear_fc2" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {"op": "sum"}
            return {"op": "concat", "dim": 0}

        gather_output = getattr(module, "gather_output", None)
        if isinstance(gather_output, bool) and not gather_output:
            return {"op": "concat", "dim": -1}

        if ".self_attention.linear_qkv" in name:
            return {"op": "concat", "dim": -1}

        if ".mlp.experts." in name:
            return {"op": "concat", "dim": 0}

        if bool(
            getattr(module, "input_is_parallel", False)
        ) and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}

        return None

    def _build_merge_hints(self, name: str, module: Any) -> dict[str, dict[str, Any]]:
        """Builds field-level tensor merge hints for one call record."""
        hints: dict[str, dict[str, Any]] = {}
        primary_output_hint = self._infer_primary_output_merge_hint(name, module)
        if primary_output_hint is not None:
            hints["primary_output"] = primary_output_hint
        if ROUTER_NAME_TOKEN in name:
            concat_dim0 = {"op": "concat", "dim": 0}
            hints["output"] = concat_dim0
            hints["router_topk_ids"] = concat_dim0
            hints["router_topk_scores"] = concat_dim0
        return hints

    def _make_hook(self, name: str, module: Any):
        def _hook(_module: Any, inputs: Any, output: Any) -> None:
            if self.current_step_index is None:
                return
            call_index = len(self.current_step_trace.get(name, []))
            trace_item: dict[str, Any] = {
                "call_index": call_index,
                "module_type": module.__class__.__name__,
                "rank_meta": _rank_metadata(),
                "merge_hints": self._build_merge_hints(name, module),
                "inputs": _materialize_trace_value(inputs),
                "output": _materialize_trace_value(output),
                "primary_input": self.guess_primary_tensor(inputs),
                "primary_output": self.guess_primary_tensor(output),
            }
            if ROUTER_NAME_TOKEN in name:
                router_topk = _extract_router_topk(output)
                if router_topk is not None:
                    topk_ids, topk_scores = router_topk
                    trace_item["router_topk_ids"] = topk_ids
                    trace_item["router_topk_scores"] = topk_scores
            self.current_step_trace.setdefault(name, []).append(trace_item)

        return _hook

    @staticmethod
    def guess_primary_tensor(value: Any) -> torch.Tensor | None:
        tensor = _extract_primary_tensor(value)
        if tensor is None:
            return None
        return _materialize_tensor(tensor)

    def set_step(self, step_index: int) -> None:
        self.current_step_index = step_index
        self.current_step_trace = {}

    @staticmethod
    def _is_moe_expert_forward_module(module_name: str) -> bool:
        """Returns whether one module emits MoE expert forward outputs."""
        if ".mlp.experts." not in module_name:
            return False
        if ".mlp.router" in module_name:
            return False
        return ".linear_fc1" in module_name or ".linear_fc2" in module_name

    @staticmethod
    def _primary_output_merge_hint(call: dict[str, Any]) -> dict[str, Any] | None:
        """Reads primary-output merge metadata from one call payload."""
        merge_hints = call.get("merge_hints")
        if not isinstance(merge_hints, dict):
            return None
        primary_hint = merge_hints.get("primary_output")
        if not isinstance(primary_hint, dict):
            return None
        return primary_hint

    @classmethod
    def _lookup_call_by_index(
        cls,
        trace: dict[str, list[dict[str, Any]]],
        module_name: str,
        call_index: int,
    ) -> dict[str, Any] | None:
        """Finds one call entry by call-index with positional fallback."""
        calls = trace.get(module_name)
        if calls is None:
            return None
        for call in calls:
            if int(call.get("call_index", -1)) == call_index:
                return call
        if 0 <= call_index < len(calls):
            return calls[call_index]
        return None

    @staticmethod
    def _router_module_name_for_expert_module(module_name: str) -> str | None:
        """Maps one expert module name to its layer router module name."""
        for token in (".mlp.experts.linear_fc1", ".mlp.experts.linear_fc2"):
            token_index = module_name.find(token)
            if token_index != -1:
                return f"{module_name[:token_index]}.mlp.router"
        return None

    @classmethod
    def _build_moe_row_identities(
        cls,
        *,
        module_name: str,
        call_index: int,
        trace: dict[str, list[dict[str, Any]]],
        row_splits: list[int] | None,
    ) -> list[tuple[int, int, int]] | None:
        """Builds stable `(expert_id, token_index, topk_slot)` identities for MoE rows."""
        router_module_name = cls._router_module_name_for_expert_module(module_name)
        if router_module_name is None:
            return None
        router_call = cls._lookup_call_by_index(trace, router_module_name, call_index)
        if router_call is None:
            return None
        router_topk_ids = router_call.get("router_topk_ids")
        if not isinstance(router_topk_ids, torch.Tensor) or router_topk_ids.ndim != 2:
            return None
        token_splits_raw = router_call.get("router_topk_ids__row_splits")
        if row_splits is None:
            if isinstance(token_splits_raw, list):
                row_splits = [
                    int(v) * int(router_topk_ids.shape[1]) for v in token_splits_raw
                ]
            else:
                row_splits = [int(router_topk_ids.numel())]
        if isinstance(token_splits_raw, list):
            token_splits = [int(v) for v in token_splits_raw]
        else:
            topk = int(router_topk_ids.shape[1])
            token_splits = [int(v) // topk for v in row_splits]
        if len(row_splits) != len(token_splits):
            return None
        row_cursor = 0
        token_cursor = 0
        identities: list[tuple[int, int, int]] = []
        for row_count, token_count in zip(row_splits, token_splits):
            local_ids = router_topk_ids[token_cursor : token_cursor + token_count]
            token_cursor += token_count
            local_identities: list[tuple[int, int, int]] = []
            max_expert = int(local_ids.max().item()) if local_ids.numel() > 0 else -1
            for expert_id in range(max_expert + 1):
                expert_rows = (local_ids == expert_id).nonzero(as_tuple=False)
                for token_offset, slot_index in expert_rows.tolist():
                    local_identities.append(
                        (expert_id, token_cursor - token_count + token_offset, slot_index)
                    )
            if len(local_identities) != row_count:
                return None
            identities.extend(local_identities)
            row_cursor += row_count
        if row_cursor != sum(row_splits):
            return None
        return identities

    @classmethod
    def _canonicalize_etp_fc1_feature_layout(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Normalizes expert-TP fc1 feature order to a topology-independent layout."""
        if ".mlp.experts.linear_fc1" not in module_name or ".lora" in module_name:
            return tensor
        if tensor.ndim != 2:
            return tensor
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "gate_up_rank_interleaved":
            return tensor
        rank_meta = call.get("rank_meta")
        etp_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                etp_world_size = first_meta.get("etp_world_size")
        elif isinstance(rank_meta, dict):
            etp_world_size = rank_meta.get("etp_world_size")
        if not isinstance(etp_world_size, int) or etp_world_size <= 1:
            return tensor
        block_count = 2 * etp_world_size
        if tensor.shape[1] % block_count != 0:
            return tensor
        blocks = torch.chunk(tensor, block_count, dim=1)
        reordered = [blocks[index] for index in range(0, block_count, 2)] + [
            blocks[index] for index in range(1, block_count, 2)
        ]
        return torch.cat(reordered, dim=1).contiguous()

    @classmethod
    def _canonicalize_moe_expert_row_order(
        cls,
        *,
        module_name: str,
        call_index: int,
        tensor: torch.Tensor,
        trace: dict[str, list[dict[str, Any]]],
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Canonicalizes MoE expert-row ordering using router replay identities."""
        if not cls._is_moe_expert_forward_module(module_name):
            return tensor
        if tensor.ndim != 2:
            return tensor
        primary_hint = cls._primary_output_merge_hint(call)
        if isinstance(primary_hint, dict) and (
            primary_hint.get("op") != "concat" or primary_hint.get("dim") != 0
        ):
            return tensor
        row_splits_raw = call.get("primary_output__row_splits")
        row_splits = (
            [int(v) for v in row_splits_raw] if isinstance(row_splits_raw, list) else None
        )
        identities = cls._build_moe_row_identities(
            module_name=module_name,
            call_index=call_index,
            trace=trace,
            row_splits=row_splits,
        )
        if identities is None or len(identities) != int(tensor.shape[0]):
            return tensor
        order = sorted(range(len(identities)), key=lambda index: identities[index])
        return tensor[order]

    @classmethod
    def _canonicalize_primary_output_tensor(
        cls,
        *,
        module_name: str,
        call_index: int,
        tensor: torch.Tensor,
        trace: dict[str, list[dict[str, Any]]],
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Runs all primary-output canonicalization passes for one call tensor."""
        tensor = cls._canonicalize_etp_fc1_feature_layout(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        return cls._canonicalize_moe_expert_row_order(
            module_name=module_name,
            call_index=call_index,
            tensor=tensor,
            trace=trace,
            call=call,
        )

    @classmethod
    def canonicalize_trace(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Canonicalizes topology-dependent trace outputs in place."""
        for module_name in sorted(trace.keys()):
            calls = trace[module_name]
            for call_offset, call in enumerate(calls):
                if bool(call.get(PRIMARY_OUTPUT_CANONICAL_KEY)):
                    continue
                call_index = int(call.get("call_index", call_offset))
                tensor = call.get("primary_output")
                if isinstance(tensor, torch.Tensor):
                    call["primary_output"] = cls._canonicalize_primary_output_tensor(
                        module_name=module_name,
                        call_index=call_index,
                        tensor=tensor,
                        trace=trace,
                        call=call,
                    )
                call[PRIMARY_OUTPUT_CANONICAL_KEY] = True
        return trace

    @classmethod
    def flatten_trace_tensors(
        cls,
        trace: dict[str, list[dict[str, Any]]],
        *,
        value_key: str,
    ) -> dict[str, Any]:
        """Flattens trace calls into deterministic key->value tensor maps."""
        if value_key == "primary_output":
            cls.canonicalize_trace(trace)
        flattened: dict[str, Any] = {}
        for module_name in sorted(trace.keys()):
            for call_offset, call in enumerate(trace[module_name]):
                tensor = call.get(value_key)
                if tensor is None:
                    continue
                call_index = call.get("call_index", call_offset)
                flattened[f"{module_name}.call_{call_index}"] = tensor
        return flattened

    @classmethod
    def _merge_rank_values(
        cls,
        values_by_rank: list[Any],
        *,
        preferred_cat_dim: int | None = None,
        preferred_reduce: str | None = None,
    ) -> Any:
        if not values_by_rank:
            raise RuntimeError("Cannot merge empty rank value list")
        if all(isinstance(value, torch.Tensor) for value in values_by_rank):
            tensors = cast(list[torch.Tensor], values_by_rank)
            if preferred_reduce == "sum" and all(
                tensors[0].shape == tensor.shape for tensor in tensors[1:]
            ):
                return torch.stack(tensors, dim=0).sum(dim=0)
            if (
                preferred_cat_dim is not None
                and all(tensor.ndim > 0 for tensor in tensors)
                and cls._can_cat_along_dim(tensors, dim=preferred_cat_dim)
            ):
                return torch.cat(tensors, dim=preferred_cat_dim)
            if all(
                tensors[0].shape == tensor.shape and torch.equal(tensors[0], tensor)
                for tensor in tensors[1:]
            ):
                return tensors[0]
            if all(tensor.ndim > 0 for tensor in tensors):
                if cls._can_cat_along_dim(tensors, dim=0):
                    return torch.cat(tensors, dim=0)
                if cls._can_cat_along_dim(tensors, dim=-1):
                    return torch.cat(tensors, dim=-1)
            if all(tensors[0].shape == tensor.shape for tensor in tensors[1:]):
                return torch.stack(tensors, dim=0)
            return tensors
        if all(isinstance(value, dict) for value in values_by_rank):
            dicts = cast(list[dict[str, Any]], values_by_rank)
            keys = sorted(set().union(*(value.keys() for value in dicts)))
            return {
                key: cls._merge_rank_values(
                    [value[key] for value in dicts if key in value],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for key in keys
            }
        if all(isinstance(value, list) for value in values_by_rank):
            lists = cast(list[list[Any]], values_by_rank)
            if any(len(values) != len(lists[0]) for values in lists[1:]):
                return lists
            return [
                cls._merge_rank_values(
                    [value[index] for value in lists],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(lists[0]))
            ]
        if all(isinstance(value, tuple) for value in values_by_rank):
            tuples = cast(list[tuple[Any, ...]], values_by_rank)
            if any(len(values) != len(tuples[0]) for values in tuples[1:]):
                return tuples
            return tuple(
                cls._merge_rank_values(
                    [value[index] for value in tuples],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(tuples[0]))
            )
        if all(value == values_by_rank[0] for value in values_by_rank[1:]):
            return values_by_rank[0]
        return values_by_rank[0]

    @classmethod
    def _merge_rank_call_entries(
        cls,
        rank_call_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merges one module call across ranks using per-field merge hints."""
        merged_call: dict[str, Any] = {}
        keys = sorted(set().union(*(entry.keys() for entry in rank_call_entries)))
        for key in keys:
            values = [entry[key] for entry in rank_call_entries if key in entry]
            if key == "rank_meta":
                merged_call[key] = values
                continue
            preferred_cat_dim: int | None = None
            preferred_reduce: str | None = None
            if values and key not in {"merge_hints", "call_index", "module_type"}:
                hint_values = [
                    cast(dict[str, Any], entry["merge_hints"]).get(key)
                    for entry in rank_call_entries
                    if isinstance(entry.get("merge_hints"), dict)
                ]
                op_hints = [
                    hint
                    for hint in hint_values
                    if isinstance(hint, dict) and isinstance(hint.get("op"), str)
                ]
                if op_hints:
                    selected_hint = op_hints[0]
                    op = selected_hint.get("op")
                    if op == "concat":
                        dim = selected_hint.get("dim")
                        if isinstance(dim, int):
                            preferred_cat_dim = dim
                    elif op == "sum":
                        preferred_reduce = "sum"
                if (
                    preferred_reduce is None
                    and preferred_cat_dim == 0
                    and all(isinstance(value, torch.Tensor) for value in values)
                ):
                    merged_call[f"{key}__row_splits"] = [
                        int(cast(torch.Tensor, value).shape[0]) for value in values
                    ]
            merged_call[key] = cls._merge_rank_values(
                values,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )
        return merged_call

    @staticmethod
    def _can_cat_along_dim(tensors: list[torch.Tensor], dim: int) -> bool:
        if not tensors:
            return False
        if tensors[0].ndim == 0:
            return False
        ndim = tensors[0].ndim
        axis = dim if dim >= 0 else ndim + dim
        if axis < 0 or axis >= ndim:
            return False
        if any(tensor.ndim != ndim for tensor in tensors[1:]):
            return False
        for dim_index in range(ndim):
            if dim_index == axis:
                continue
            dim_size = tensors[0].shape[dim_index]
            if any(tensor.shape[dim_index] != dim_size for tensor in tensors[1:]):
                return False
        return True

    @classmethod
    def _merge_rank_traces(
        cls,
        rank_traces: list[dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        if len(rank_traces) == 1:
            return rank_traces[0]
        merged: dict[str, list[dict[str, Any]]] = {}
        module_names = sorted(set().union(*(trace.keys() for trace in rank_traces)))
        for module_name in module_names:
            call_count = max(len(trace.get(module_name, [])) for trace in rank_traces)
            module_calls: list[dict[str, Any]] = []
            for call_index in range(call_count):
                rank_values = [
                    trace[module_name][call_index]
                    for trace in rank_traces
                    if module_name in trace and call_index < len(trace[module_name])
                ]
                if not rank_values:
                    continue
                module_calls.append(cls._merge_rank_call_entries(rank_values))
            merged[module_name] = module_calls
        return merged

    @staticmethod
    def _gather_rank_traces(
        local_trace: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, list[dict[str, Any]]]] | None:
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return [local_trace]
        gathered: list[dict[str, list[dict[str, Any]]] | None] = [
            None
        ] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local_trace)
        if torch.distributed.get_rank() != 0:
            return None
        return cast(list[dict[str, list[dict[str, Any]]]], gathered)

    def save_current_step(self, traces_dir: Path) -> Path | None:
        if not self.enabled or self.current_step_index is None:
            return None
        gathered_traces = self._gather_rank_traces(self.current_step_trace)
        if gathered_traces is None:
            return None
        merged_trace = self.canonicalize_trace(self._merge_rank_traces(gathered_traces))
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_path = traces_dir / f"forward_trace_step_{self.current_step_index:03d}.pt"
        torch.save(merged_trace, trace_path)
        return trace_path

    @classmethod
    def load_trace(cls, trace_path: Path) -> dict[str, list[dict[str, Any]]]:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        return cls.canonicalize_trace(trace)

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
