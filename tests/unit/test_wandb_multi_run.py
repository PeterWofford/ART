import os
import sys
from pathlib import Path
from unittest.mock import patch

from art import Model


def test_wandb_creates_separate_runs_per_model(tmp_path: Path):
    class FakeRun:
        def __init__(self, name: str):
            self.name = name
            self.id = name
            self._is_finished = False
            self.defined_metrics: list[tuple[str, str | None]] = []

        def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
            self.defined_metrics.append((name, step_metric))

    class FakeWandb:
        def __init__(self):
            self.init_calls: list[dict] = []
            self.runs: list[FakeRun] = []

        @staticmethod
        def Settings(**kwargs):
            return kwargs

        def init(self, **kwargs):
            self.init_calls.append(kwargs)
            run = FakeRun(kwargs["name"])
            self.runs.append(run)
            return run

        def define_metric(self, *args, **kwargs) -> None:
            raise AssertionError("Model should define metrics on the run object")

    fake_wandb = FakeWandb()
    model_one = Model(
        name="run-one",
        project="test-project",
        base_path=str(tmp_path),
    )
    model_two = Model(
        name="run-two",
        project="test-project",
        base_path=str(tmp_path),
    )

    with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            run_one = model_one._get_wandb_run()
            run_two = model_two._get_wandb_run()
            model_one._define_wandb_step_metrics(["costs/train/custom"])

    assert run_one is not None
    assert run_two is not None
    assert run_one is not run_two
    assert [call["name"] for call in fake_wandb.init_calls] == [
        "run-one",
        "run-two",
    ]
    assert all(call["reinit"] == "create_new" for call in fake_wandb.init_calls)
    assert run_one.defined_metrics == [
        ("training_step", None),
        ("time/wall_clock_sec", None),
        ("reward/*", "training_step"),
        ("loss/*", "training_step"),
        ("throughput/*", "training_step"),
        ("costs/*", "training_step"),
        ("time/*", "training_step"),
        ("data/*", "training_step"),
        ("train/*", "training_step"),
        ("val/*", "training_step"),
        ("test/*", "training_step"),
        ("discarded/*", "training_step"),
        ("costs/train/custom", "training_step"),
    ]
    assert run_two.defined_metrics == [
        ("training_step", None),
        ("time/wall_clock_sec", None),
        ("reward/*", "training_step"),
        ("loss/*", "training_step"),
        ("throughput/*", "training_step"),
        ("costs/*", "training_step"),
        ("time/*", "training_step"),
        ("data/*", "training_step"),
        ("train/*", "training_step"),
        ("val/*", "training_step"),
        ("test/*", "training_step"),
        ("discarded/*", "training_step"),
    ]
