from __future__ import annotations

from typing import Any, cast

from quack.gemm import gemm as quack_gemm
import torch

_PADDED_LOW_RANK_TARGET = 8
_PADDED_LOW_RANKS = frozenset({1, 2, 4})
_SUPPORTED_RANKS = frozenset({1, 2, 4, 8, 16, 32, 64, 128})


def _validate_supported_rank(rank: int) -> None:
    if rank not in _SUPPORTED_RANKS:
        raise ValueError(
            f"Grouped LoRA QuACK backend only supports ranks {sorted(_SUPPORTED_RANKS)}, got {rank}"
        )


def _tokens_per_expert_to_tensor(
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(tokens_per_expert, list):
        return torch.tensor(tokens_per_expert, dtype=torch.int64, device="cpu")
    if tokens_per_expert.ndim != 1:
        raise ValueError(
            f"tokens_per_expert must be 1D, got shape {tuple(tokens_per_expert.shape)}"
        )
    return tokens_per_expert.detach().to(dtype=torch.int64).contiguous()


def _build_expert_offsets(
    tokens_per_expert: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.empty(
        tokens_per_expert.numel() + 1,
        dtype=torch.int64,
        device=tokens_per_expert.device,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)
    return offsets.to(device=device, dtype=torch.int32)


def _validate_inputs(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    counts = _tokens_per_expert_to_tensor(tokens_per_expert)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if a_t.ndim != 3:
        raise ValueError(f"a_t must be 3D, got shape {tuple(a_t.shape)}")
    if b_t.ndim != 3:
        raise ValueError(f"b_t must be 3D, got shape {tuple(b_t.shape)}")
    rank = a_t.shape[-1]
    _validate_supported_rank(rank)
    if b_t.shape[-2] != rank:
        raise ValueError(f"Expected b_t rank dim {rank}, got shape {tuple(b_t.shape)}")
    if a_t.shape[0] != b_t.shape[0]:
        raise ValueError(
            "a_t and b_t must have the same number of experts, "
            f"got {a_t.shape[0]} and {b_t.shape[0]}"
        )
    if a_t.shape[1] != x.shape[1]:
        raise ValueError(
            f"a_t input dim must match x.shape[1], got {a_t.shape[1]} and {x.shape[1]}"
        )
    if counts.numel() != a_t.shape[0]:
        raise ValueError(
            "tokens_per_expert length must match number of experts, "
            f"got {counts.numel()} and {a_t.shape[0]}"
        )
    if x.device.type != "cuda" or a_t.device != x.device or b_t.device != x.device:
        raise ValueError("x, a_t, and b_t must be CUDA tensors on the same device")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"Unsupported dtype {x.dtype}; expected fp16 or bf16")
    if a_t.dtype != x.dtype or b_t.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: x={x.dtype}, a_t={a_t.dtype}, b_t={b_t.dtype}"
        )
    return counts


def _effective_rank(rank: int) -> int:
    if rank in _PADDED_LOW_RANKS:
        return _PADDED_LOW_RANK_TARGET
    return rank


def _pad_a_t(a_t: torch.Tensor, effective_rank: int) -> torch.Tensor:
    pad_rank = effective_rank - a_t.shape[-1]
    if pad_rank <= 0:
        return a_t.contiguous()
    pad = a_t.new_zeros(a_t.shape[0], a_t.shape[1], pad_rank)
    return torch.cat((a_t, pad), dim=-1).contiguous()


def _pad_b_t(b_t: torch.Tensor, effective_rank: int) -> torch.Tensor:
    pad_rank = effective_rank - b_t.shape[1]
    if pad_rank <= 0:
        return b_t.contiguous()
    pad = b_t.new_zeros(b_t.shape[0], pad_rank, b_t.shape[2])
    return torch.cat((b_t, pad), dim=1).contiguous()


def _proj_tile_n(rank: int) -> int:
    return 64 if rank <= 64 else 128


def _matmul_tile_n(out_features: int) -> int:
    return 128 if out_features >= 128 else 64


def _grad_a_tile_m(rank: int) -> int:
    return 128


def _grad_b_tile_m(rank: int) -> int:
    return 64 if rank <= 64 else 128


def _varlen_quack_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_features: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
    tile_n: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    out = torch.empty(
        a.shape[0],
        out_features,
        device=a.device,
        dtype=a.dtype,
    )
    quack_gemm(
        a,
        b,
        out,
        None,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        alpha=alpha,
        cu_seqlens_m=expert_offsets,
    )
    return out


def _varlen_quack_gemm_k(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    batch_count: int,
    out_shape_m: int,
    out_shape_n: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
    tile_n: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    out = torch.empty(
        batch_count,
        out_shape_m,
        out_shape_n,
        device=a.device,
        dtype=a.dtype,
    )
    quack_gemm(
        a,
        b,
        out,
        None,
        None,
        tile_M=tile_m,
        tile_N=tile_n,
        cluster_M=1,
        cluster_N=1,
        persistent=True,
        alpha=alpha,
        cu_seqlens_k=expert_offsets,
    )
    return out


class _QuackGroupedLoraFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        counts: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        expert_offsets = _build_expert_offsets(counts, device=x.device)
        actual_rank = a_t.shape[-1]
        effective_rank = _effective_rank(actual_rank)
        a_t_eff = _pad_a_t(a_t, effective_rank)
        b_t_eff = _pad_b_t(b_t, effective_rank)
        proj_weights = a_t_eff.permute(0, 2, 1).contiguous()
        apply_weights = b_t_eff.permute(0, 2, 1).contiguous()

        tmp = _varlen_quack_gemm(
            x.contiguous(),
            proj_weights,
            out_features=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(effective_rank),
        )
        out = _varlen_quack_gemm(
            tmp,
            apply_weights,
            out_features=b_t.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(b_t.shape[-1]),
            alpha=scale,
        )

        ctx.save_for_backward(x, a_t_eff, b_t_eff, tmp, expert_offsets)
        ctx.actual_rank = actual_rank
        ctx.effective_rank = effective_rank
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        if len(grad_outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one gradient output, got {len(grad_outputs)}"
            )
        x, a_t_eff, b_t_eff, tmp, expert_offsets = ctx.saved_tensors
        effective_rank = ctx.effective_rank
        actual_rank = ctx.actual_rank
        scale = ctx.scale
        grad_out = cast(torch.Tensor, grad_outputs[0])
        assert grad_out.stride(-1) == 1, (
            "QuACK grouped LoRA backward requires grad_out stride(-1) == 1"
        )

        grad_tmp = _varlen_quack_gemm(
            grad_out,
            b_t_eff.contiguous(),
            out_features=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_proj_tile_n(effective_rank),
            alpha=scale,
        )
        grad_x = _varlen_quack_gemm(
            grad_tmp,
            a_t_eff.contiguous(),
            out_features=x.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=64,
            tile_n=_matmul_tile_n(x.shape[-1]),
        )
        grad_a_eff = _varlen_quack_gemm_k(
            x.transpose(0, 1),
            grad_tmp.transpose(0, 1),
            batch_count=a_t_eff.shape[0],
            out_shape_m=a_t_eff.shape[1],
            out_shape_n=effective_rank,
            expert_offsets=expert_offsets,
            tile_m=_grad_a_tile_m(effective_rank),
            tile_n=_proj_tile_n(effective_rank),
        )
        grad_b_eff = _varlen_quack_gemm_k(
            tmp.transpose(0, 1),
            grad_out.transpose(0, 1),
            batch_count=b_t_eff.shape[0],
            out_shape_m=effective_rank,
            out_shape_n=b_t_eff.shape[-1],
            expert_offsets=expert_offsets,
            tile_m=_grad_b_tile_m(effective_rank),
            tile_n=_matmul_tile_n(b_t_eff.shape[-1]),
            alpha=scale,
        )
        return (
            grad_x,
            grad_a_eff[:, :, :actual_rank].contiguous(),
            grad_b_eff[:, :actual_rank, :].contiguous(),
            None,
            None,
        )


def quack_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: list[int] | torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Run grouped LoRA with the QuACK varlen GEMM backend.

    Assumptions required by the caller:
    - `counts` is ordered by local expert index and `sum(counts) == x.shape[0]`.
    - `counts` length matches `a_t.shape[0] == b_t.shape[0]`.
    - `x.shape[1] == a_t.shape[1]` and `a_t.shape[-1] == b_t.shape[-2]`.
    - `x`, `a_t`, and `b_t` are CUDA tensors on the same device with fp16 or bf16 dtype.

    The value-based `sum(counts)` check is intentionally omitted to avoid a host-device
    synchronization in the hot path.
    """
    counts_tensor = _validate_inputs(x, a_t, b_t, counts)
    return _QuackGroupedLoraFn.apply(x, a_t, b_t, counts_tensor, scale)
