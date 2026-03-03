from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art.model import TrainableModel
from art.trajectories import Trajectory
from art.types import TrainSFTConfig


@pytest.mark.asyncio
async def test_serverless_train_sft_records_provenance_on_training_end():
    """Serverless SFT should append serverless-sft provenance on completion."""
    from art.serverless.backend import ServerlessBackend

    mock_client = MagicMock()
    mock_client.base_url = "https://api.training.wandb.ai/v1"
    mock_client.api_key = "test-key"
    mock_client.sft_training_jobs = MagicMock()
    mock_client.sft_training_jobs.create = AsyncMock(
        return_value=SimpleNamespace(id="job-123")
    )

    async def list_events(*args, **kwargs):
        yield SimpleNamespace(
            id="evt-1",
            type="training_started",
            data={"num_sequences": 1},
        )
        yield SimpleNamespace(
            id="evt-2",
            type="training_ended",
            data={},
        )

    mock_client.sft_training_jobs.events = MagicMock()
    mock_client.sft_training_jobs.events.list = list_events

    fake_logged_artifact = MagicMock()
    fake_logged_artifact.wait.return_value = fake_logged_artifact

    fake_wandb_run = MagicMock()
    fake_wandb_run.log_artifact.return_value = fake_logged_artifact

    with (
        patch("art.serverless.backend.Client", return_value=mock_client),
        patch("art.serverless.backend.asyncio.sleep", new=AsyncMock()),
        patch("art.serverless.backend.record_provenance") as mock_record_provenance,
        patch("wandb.init", return_value=fake_wandb_run),
        patch("wandb.Artifact", return_value=MagicMock()),
    ):
        backend = ServerlessBackend(api_key="test-key")

        model = TrainableModel(
            name="provenance-sft-test",
            project="test-project",
            base_model="OpenPipe/Qwen3-14B-Instruct",
        )
        model.id = "model-123"
        model.entity = "test-entity"
        model.run_id = "test-run-id"

        fake_model_run = MagicMock()

        trajectories = [
            Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "Say hello"},
                    {"role": "assistant", "content": "hello"},
                ],
                reward=1.0,
            )
        ]
        config = TrainSFTConfig(learning_rate=1e-5, batch_size=1)

        with patch.object(model, "_get_wandb_run", return_value=fake_model_run):
            results = [
                metrics
                async for metrics in backend._train_sft(
                    model=model,
                    trajectories=trajectories,
                    config=config,
                    dev_config={},
                    verbose=False,
                )
            ]

        assert results == []
        mock_record_provenance.assert_called_once_with(
            fake_model_run, "serverless-sft"
        )
