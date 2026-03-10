import pytest

from .megatron_oracle_harness import (
    ORACLE_TOPOLOGY,
    PHASE_A_TOPOLOGIES,
    PHASE_B_TOPOLOGIES,
    SENSITIVITY_MUTATION_ENV,
    SENSITIVITY_TOPOLOGY,
    available_gpu_count,
    default_case_config,
    ensure_oracle_reference_artifacts,
    phase_b_dp_enabled,
    regenerate_requested,
    run_and_compare_topology,
    run_sensitivity_check,
    sensitivity_enabled,
)


def _require_gpus_for(topology_world_size: int) -> None:
    gpu_count = available_gpu_count()
    if gpu_count < topology_world_size:
        pytest.skip(
            f"Need {topology_world_size} GPUs for topology run, only found {gpu_count}"
        )


def _skip_if_sensitivity_mode() -> None:
    if sensitivity_enabled():
        pytest.skip(
            f"{SENSITIVITY_MUTATION_ENV} is enabled; running sensitivity check only."
        )


def _run_topology_case(  # type: ignore[no-untyped-def]
    topology,
    case_config,
    *,
    regenerate: bool,
) -> None:
    _require_gpus_for(topology.world_size())
    run_and_compare_topology(
        case_config=case_config,
        topology=topology,
        regenerate=regenerate,
    )


def test_000_megatron_lora_oracle_sensitivity_check() -> None:
    if not sensitivity_enabled():
        pytest.skip(
            f"Set {SENSITIVITY_MUTATION_ENV}=drop_finalize to enable sensitivity check."
        )
    _require_gpus_for(SENSITIVITY_TOPOLOGY.world_size())
    run_sensitivity_check(
        case_config=default_case_config(),
        regenerate=regenerate_requested(),
    )


def test_megatron_lora_oracle_phase_a_matrix() -> None:
    _skip_if_sensitivity_mode()
    case_config = default_case_config()
    regenerate = regenerate_requested()
    _require_gpus_for(ORACLE_TOPOLOGY.world_size())
    ensure_oracle_reference_artifacts(
        case_config=case_config,
        regenerate=regenerate,
    )
    for topology in PHASE_A_TOPOLOGIES:
        _run_topology_case(
            topology,
            case_config,
            regenerate=regenerate and topology.slug() != ORACLE_TOPOLOGY.slug(),
        )


@pytest.mark.parametrize(
    "topology_index",
    range(len(PHASE_B_TOPOLOGIES)),
    ids=[topology.slug() for topology in PHASE_B_TOPOLOGIES],
)
def test_megatron_lora_oracle_phase_b_dp_matrix(topology_index: int) -> None:
    _skip_if_sensitivity_mode()
    if not phase_b_dp_enabled():
        pytest.xfail(
            "DP matrix currently blocked until Megatron backend DP support is enabled"
        )
    case_config = default_case_config()
    regenerate = regenerate_requested()
    _require_gpus_for(ORACLE_TOPOLOGY.world_size())
    ensure_oracle_reference_artifacts(
        case_config=case_config,
        regenerate=regenerate,
    )
    _run_topology_case(
        PHASE_B_TOPOLOGIES[topology_index],
        case_config,
        regenerate=regenerate,
    )
