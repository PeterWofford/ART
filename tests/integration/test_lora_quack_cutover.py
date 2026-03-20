from __future__ import annotations

import pytest
import torch

pytest.importorskip("quack")

from art.megatron.lora import LoRA


def _eager_grouped_lora(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, token_count in enumerate(counts.tolist()):
        if token_count == 0:
            continue
        stop = start + int(token_count)
        outputs.append(x[start:stop] @ a_t[expert_idx] @ b_t[expert_idx])
        start = stop
    if start != x.shape[0]:
        raise RuntimeError(
            f"Grouped split mismatch: consumed {start} rows for shape {tuple(x.shape)}"
        )
    return torch.cat(outputs, dim=0) * scale


@pytest.mark.parametrize("rank", [1, 4, 16])
def test_lora_grouped_forward_cutover_matches_reference(rank: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the LoRA QuACK cutover test.")

    device = torch.device("cuda:0")
    torch.manual_seed(20260323 + rank)

    lora = LoRA(
        adapter_model_prefix="test.{expert}",
        in_features=64,
        out_features=64,
        rank=rank,
        alpha=32,
        dtype=torch.bfloat16,
        device=device,
        num_local_experts=4,
    )
    with torch.no_grad():
        lora.A_T.copy_(torch.randn_like(lora.A_T) * 0.05)
        lora.B_T.copy_(torch.randn_like(lora.B_T) * 0.05)

    counts = torch.tensor([32, 0, 16, 24], dtype=torch.int64)
    total_tokens = int(counts.sum().item())
    x = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16) * 0.05
    loss_grad = torch.randn(total_tokens, 64, device=device, dtype=torch.bfloat16)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = lora.A_T.detach().clone().requires_grad_(True)
    b_ref = lora.B_T.detach().clone().requires_grad_(True)
    ref_out = _eager_grouped_lora(
        x_ref,
        a_ref,
        b_ref,
        counts,
        scale=lora.scale,
    )
    ref_loss = (ref_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    ref_loss.backward()

    x_test = x.detach().clone().requires_grad_(True)
    lora.zero_grad(set_to_none=True)
    got_out = lora(x_test, tokens_per_expert=counts)
    got_loss = (got_out.float() * loss_grad.float()).sum() / max(1, loss_grad.numel())
    got_loss.backward()

    assert torch.allclose(ref_out, got_out.detach(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(x_ref.grad, x_test.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(a_ref.grad, lora.A_T.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(b_ref.grad, lora.B_T.grad, atol=5e-2, rtol=5e-2)
