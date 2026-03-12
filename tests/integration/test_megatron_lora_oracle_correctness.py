import pytest

from .megatron_oracle_harness import (
    EXTENDED_TOPOLOGIES,
    SENSITIVITY_MUTATION_ENV,
    SENSITIVITY_TOPOLOGY,
    TOPOLOGIES,
    available_gpu_count,
    case_config,
    extended_topologies_enabled,
    run_sensitivity_suite,
    run_suite,
    sensitivity_enabled,
    sensitivity_mutations,
)


def _require_gpus_for(topology_world_size: int) -> None:
    gpu_count = available_gpu_count()
    if gpu_count < topology_world_size:
        pytest.skip(
            f"Need {topology_world_size} GPUs for topology run, only found {gpu_count}"
        )


def _suite_world_size() -> int:
    suite_topologies = list(TOPOLOGIES)
    if extended_topologies_enabled():
        suite_topologies.extend(EXTENDED_TOPOLOGIES)
    return max(topology.world_size() for topology in suite_topologies)


def test_megatron_lora_diff_sensitivity() -> None:
    """
    Runs a each of the sensitivity mutations (e.g. drop megatron finalize grads)
    and expects each to fail (numerical differences larger than our thresholds)

    This test ensures we can catch errors we know of (implying we will be able to catch unknown errors as well)
    """
    if not sensitivity_enabled():
        pytest.skip(
            f"Set {SENSITIVITY_MUTATION_ENV}=drop_finalize (or CSV) to enable sensitivity check."
        )
    _require_gpus_for(SENSITIVITY_TOPOLOGY.world_size())
    mutations = sensitivity_mutations()
    assert mutations
    run_sensitivity_suite(
        case_config=case_config(),
        mutations=mutations,
    )


def test_megatron_lora_topology_suite() -> None:
    """
    Runs the suite of topologies and expects each to pass (numerical differences within our thresholds)
    """
    _require_gpus_for(_suite_world_size())
    run_suite(
        case_config=case_config(),
    )
