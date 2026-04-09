from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from openai.types.chat.chat_completion import Choice

pytest.importorskip("torch")

import art
from art import TrainableModel, Trajectory, TrajectoryGroup
from art.local import LocalBackend


ART_ROOT = Path(__file__).resolve().parents[2]
for path in (ART_ROOT, ART_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import dev.train_ruler as legacy_ruler  # noqa: E402
import dev.train_ruler_pipeline as pilot_ruler  # noqa: E402


class _ContinueFinalMessageRejectingTokenizer:
    """Minimal tokenizer that reproduces the HF chat-template failure.

    The real failure comes from HF apply_chat_template(..., continue_final_message=True)
    rejecting a final assistant message that disappears under the chat template.
    For a dry unit test, we model that directly when the final assistant turn carries
    tool calls, which is the same shape our IVR rollouts produce.
    """

    chat_template = ""
    vocab_size = 512
    eos_token = "\x00"
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools=None,
        tokenize: bool = True,
        return_dict=None,
        continue_final_message: bool = False,
        **kwargs,
    ):
        del tools, kwargs
        if (
            continue_final_message
            and messages
            and messages[-1].get("role") == "assistant"
            and messages[-1].get("tool_calls")
        ):
            raise ValueError(
                "continue_final_message is set but the final message does not appear "
                "in the chat after applying the chat template!"
            )

        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}" for message in messages
        )
        if not tokenize:
            return rendered

        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return chr(token_ids)
        return "".join(chr(token_id) for token_id in token_ids)


def _tool_choice() -> Choice:
    return Choice.model_validate(
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": "x",
                        "bytes": [120],
                        "logprob": -0.1,
                        "top_logprobs": [],
                    }
                ],
                "refusal": None,
            },
            "message": {
                "content": "",
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [
                    {
                        "id": "call_wait_1",
                        "function": {
                            "arguments": "{}",
                            "name": "wait",
                        },
                        "type": "function",
                    }
                ],
            },
        }
    )


def _sample_train_row() -> dict[str, Any]:
    return {
        "input": {
            "messages": [
                {"role": "system", "content": "You are navigating an IVR."},
                {
                    "role": "user",
                    "content": "The IVR is still speaking. Choose the next tool.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "wait",
                        "description": "Wait for more IVR audio.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
        }
    }


def _build_group(module: Any) -> TrajectoryGroup:
    row = _sample_train_row()
    messages = row["input"]["messages"]
    tools = row["input"]["tools"]
    context, _ = module.split_context_and_golden(messages)
    choice = _tool_choice()

    return art.TrajectoryGroup(
        [
            Trajectory(
                messages_and_choices=context + [choice],
                tools=tools,
                reward=1.0,
            ),
            Trajectory(
                messages_and_choices=context + [choice.model_copy(deep=True)],
                tools=tools,
                reward=0.0,
            ),
        ]
    )


def test_legacy_and_pilot_build_same_bad_trajectory_shape() -> None:
    legacy_group = _build_group(legacy_ruler)
    pilot_group = _build_group(pilot_ruler)

    assert len(legacy_group.trajectories) == len(pilot_group.trajectories) == 2
    assert [trajectory.messages() for trajectory in legacy_group.trajectories] == [
        trajectory.messages() for trajectory in pilot_group.trajectories
    ]


@pytest.mark.parametrize(
    "module",
    [legacy_ruler, pilot_ruler],
    ids=["legacy", "pilot"],
)
def test_both_ivr_paths_hit_same_continue_final_message_failure(
    tmp_path: Path,
    module: Any,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name=f"continue-final-message-{module.__name__.split('.')[-1]}",
        project="ivr-nav-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend._tokenizers[model.base_model] = _ContinueFinalMessageRejectingTokenizer()
    backend._image_processors[model.base_model] = None

    with pytest.raises(ValueError, match="continue_final_message is set"):
        backend._get_packed_tensors(
            model,
            [_build_group(module)],
            advantage_balance=0.0,
            allow_training_without_logprobs=False,
            scale_rewards=True,
            plot_tensors=False,
        )
