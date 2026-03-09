import pytest

from art.metrics_taxonomy import (
    TrajectoryBatchSummary,
    average_metric_samples,
    build_training_summary_metrics,
)


def test_average_metric_samples_handles_sparse_keys() -> None:
    averaged = average_metric_samples(
        [
            {"loss/train": 1.0, "loss/grad_norm": 0.5},
            {"loss/train": 0.5},
            {"loss/grad_norm": 1.0},
        ]
    )

    assert averaged["loss/train"] == pytest.approx(0.75)
    assert averaged["loss/grad_norm"] == pytest.approx(0.75)


def test_build_training_summary_metrics_includes_data_and_train_sections() -> None:
    summary = TrajectoryBatchSummary(
        num_scenarios=2,
        num_trajectories=5,
        num_groups_submitted=2,
        num_groups_trainable=1,
        scenario_ids=["a", "b"],
    )

    metrics = build_training_summary_metrics(
        summary,
        include_trainable_groups=True,
    )

    assert metrics["data/step_num_scenarios"] == pytest.approx(2.0)
    assert metrics["data/step_num_groups_trainable"] == pytest.approx(1.0)
    assert metrics["train/num_groups_submitted"] == pytest.approx(2.0)
    assert metrics["train/num_trajectories"] == pytest.approx(5.0)
