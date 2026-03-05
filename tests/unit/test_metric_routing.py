import json
import os
from pathlib import Path
import types
from unittest.mock import MagicMock, patch

from art import Model


class TestMetricRoutingBaseline:
    def test_log_metrics_routes_known_sections_without_split_prefix(
        self, tmp_path: Path
    ) -> None:
        model = Model(
            name="test-model",
            project="test-project",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        model._log_metrics(
            {
                "reward/mean": 0.9,
                "custom": 1.0,
                "rewardish/value": 2.0,
            },
            split="train",
            step=7,
        )

        history_path = tmp_path / "test-project/models/test-model/history.jsonl"
        with open(history_path) as f:
            entry = json.loads(f.readline())

        assert entry["reward/mean"] == 0.9
        assert entry["train/custom"] == 1.0
        assert entry["train/rewardish/value"] == 2.0
        assert entry["training_step"] == 7
        assert entry["time/wall_clock_sec"] >= 0

    def test_get_wandb_run_registers_taxonomy_sections(self, tmp_path: Path) -> None:
        fake_run = MagicMock()
        fake_run._is_finished = False

        fake_wandb = types.SimpleNamespace()
        fake_wandb.init = MagicMock(return_value=fake_run)
        fake_wandb.define_metric = MagicMock()
        fake_wandb.Settings = lambda **kwargs: kwargs

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}, clear=False):
            with patch.dict("sys.modules", {"wandb": fake_wandb}):
                model = Model(
                    name="test-model",
                    project="test-project",
                    base_path=str(tmp_path),
                )
                run = model._get_wandb_run()

        assert run is fake_run
        define_calls = [
            (call.args, call.kwargs)
            for call in fake_wandb.define_metric.call_args_list
        ]
        assert define_calls == [
            (("training_step",), {}),
            (("time/wall_clock_sec",), {}),
            (("reward/*",), {"step_metric": "training_step"}),
            (("loss/*",), {"step_metric": "training_step"}),
            (("throughput/*",), {"step_metric": "training_step"}),
            (("costs/*",), {"step_metric": "training_step"}),
            (("time/*",), {"step_metric": "training_step"}),
            (("data/*",), {"step_metric": "training_step"}),
            (("train/*",), {"step_metric": "training_step"}),
            (("val/*",), {"step_metric": "training_step"}),
            (("test/*",), {"step_metric": "training_step"}),
        ]
