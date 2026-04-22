from __future__ import annotations

import ast
from pathlib import Path
import re
import statistics
from typing import Any

import pytest

ART_ROOT = Path(__file__).resolve().parents[2]
TRAIN_RULER_PATH = ART_ROOT / "dev" / "train_ruler.py"

_FUNCTION_NAMES = {
    "flatten_message_content",
    "normalize_text",
    "extract_user_texts",
    "count_pattern_hits",
    "classify_eval_row",
    "percentile",
    "safe_rate",
    "is_say_action",
    "summarize_holdout_failure_modes",
    "summarize_test_failure_modes",
}


def _load_metric_helpers() -> dict[str, Any]:
    source = TRAIN_RULER_PATH.read_text()
    tree = ast.parse(source, filename=str(TRAIN_RULER_PATH))
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTION_NAMES
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: dict[str, Any] = {
        "Any": Any,
        "re": re,
        "statistics": statistics,
        "ALIGNED_TIE_EPSILON": 0.03,
    }
    exec(compile(module, str(TRAIN_RULER_PATH), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def metric_helpers() -> dict[str, Any]:
    return _load_metric_helpers()


def test_classify_eval_row_detects_actionable_and_extractable_states(
    metric_helpers: dict[str, Any],
) -> None:
    classify_eval_row = metric_helpers["classify_eval_row"]

    keypad_row = classify_eval_row(
        messages=[
            {
                "role": "user",
                "content": 'Message from IVR stream: "using your keypad, enter your account number or social security number."',
            }
        ],
        tools=[],
    )
    assert keypad_row["keypad_entry"] is True
    assert keypad_row["actionable_now"] is True
    assert keypad_row["extractable_now"] is False

    extractable_row = classify_eval_row(
        messages=[
            {
                "role": "user",
                "content": 'Message from IVR stream: "your current balance is $694.15."',
            },
            {
                "role": "user",
                "content": (
                    'Message from IVR stream: "your last payment of $33.61 was received '
                    "on december 12, 2025. there's no payment due at this time.\""
                ),
            },
        ],
        tools=[],
    )
    assert extractable_row["extractable_now"] is True
    assert extractable_row["actionable_now"] is False


def test_summarize_holdout_failure_modes_tracks_model_and_gpt41_patterns(
    metric_helpers: dict[str, Any],
) -> None:
    summarize_holdout_failure_modes = metric_helpers["summarize_holdout_failure_modes"]

    holdout_results = [
        {
            "score_delta": -0.35,
            "model_score": 0.10,
            "gpt41_score": 0.45,
            "model_action": "wait",
            "gpt41_action": "enter_digit",
            "actionable_now": True,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
            "likely_stuck_loop": False,
        },
        {
            "score_delta": -0.54,
            "model_score": 0.45,
            "gpt41_score": 0.99,
            "model_action": "wait",
            "gpt41_action": "hangup_and_extract",
            "actionable_now": False,
            "extractable_now": True,
            "keypad_entry": False,
            "yes_no_prompt": False,
            "likely_stuck_loop": False,
        },
        {
            "score_delta": 0.40,
            "model_score": 0.50,
            "gpt41_score": 0.10,
            "model_action": "say_account_information",
            "gpt41_action": "wait",
            "actionable_now": True,
            "extractable_now": False,
            "keypad_entry": True,
            "yes_no_prompt": False,
            "likely_stuck_loop": False,
        },
        {
            "score_delta": -0.50,
            "model_score": 0.20,
            "gpt41_score": 0.70,
            "model_action": "say_no",
            "gpt41_action": "wait",
            "actionable_now": False,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
            "likely_stuck_loop": False,
        },
        {
            "score_delta": -0.20,
            "model_score": 0.40,
            "gpt41_score": 0.60,
            "model_action": "navigation_failed",
            "gpt41_action": "wait",
            "actionable_now": False,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
            "likely_stuck_loop": True,
        },
        {
            "score_delta": -0.15,
            "model_score": 0.10,
            "gpt41_score": 0.25,
            "model_action": "hangup_and_extract",
            "gpt41_action": "wait",
            "actionable_now": False,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
            "likely_stuck_loop": False,
        },
    ]

    metrics = summarize_holdout_failure_modes(holdout_results)

    assert metrics["support/actionable_rows"] == pytest.approx(2.0)
    assert metrics["support/extractable_rows"] == pytest.approx(1.0)
    assert metrics["support/keypad_rows"] == pytest.approx(1.0)
    assert metrics["support/stuck_loop_rows"] == pytest.approx(1.0)

    assert metrics["action_rate/model/wait"] == pytest.approx(2 / 6)
    assert metrics["action_rate/gpt41/wait"] == pytest.approx(4 / 6)

    assert metrics["failure/model/wait_on_actionable_rate"] == pytest.approx(0.5)
    assert metrics["failure/model/wait_after_completion_rate"] == pytest.approx(1.0)
    assert metrics["failure/model/missed_extract_rate"] == pytest.approx(1.0)
    assert metrics["failure/model/premature_extract_rate"] == pytest.approx(0.2)
    assert metrics["failure/model/speak_on_keypad_rate"] == pytest.approx(1.0)
    assert metrics["failure/model/say_no_misuse_rate"] == pytest.approx(1 / 6)
    assert metrics["failure/model/navigation_failed_too_early_rate"] == pytest.approx(
        0.0
    )
    assert metrics["failure/model/wait_loss_rate"] == pytest.approx(1.0)

    assert metrics["failure/gpt41/wait_on_actionable_rate"] == pytest.approx(0.5)
    assert metrics["failure/gpt41/missed_extract_rate"] == pytest.approx(0.0)
    assert metrics["failure/gpt41/wait_loss_rate"] == pytest.approx(0.25)

    assert metrics["failure/compare/model_wait_vs_gpt41_action_rate"] == pytest.approx(
        2 / 6
    )
    assert metrics["failure/compare/gpt41_wait_vs_model_action_rate"] == pytest.approx(
        4 / 6
    )
    assert metrics["compare/score_delta_mean"] == pytest.approx(
        statistics.mean(float(r["score_delta"]) for r in holdout_results)
    )


def test_summarize_test_failure_modes_breaks_down_wrong_tool_vs_wrong_args(
    metric_helpers: dict[str, Any],
) -> None:
    summarize_test_failure_modes = metric_helpers["summarize_test_failure_modes"]

    test_results = [
        {
            "match": False,
            "model_action": "wait",
            "golden_action": "enter_digit",
            "actionable_now": True,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
        },
        {
            "match": False,
            "model_action": "say_account_information",
            "golden_action": "enter_digit",
            "actionable_now": True,
            "extractable_now": False,
            "keypad_entry": True,
            "yes_no_prompt": False,
        },
        {
            "match": False,
            "model_action": "enter_digit",
            "golden_action": "enter_digit",
            "actionable_now": True,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
        },
        {
            "match": False,
            "model_action": "say_no",
            "golden_action": "wait",
            "actionable_now": False,
            "extractable_now": False,
            "keypad_entry": False,
            "yes_no_prompt": False,
        },
        {
            "match": False,
            "model_action": "wait",
            "golden_action": "hangup_and_extract",
            "actionable_now": False,
            "extractable_now": True,
            "keypad_entry": False,
            "yes_no_prompt": False,
        },
        {
            "match": True,
            "model_action": "hangup_and_extract",
            "golden_action": "hangup_and_extract",
            "actionable_now": False,
            "extractable_now": True,
            "keypad_entry": False,
            "yes_no_prompt": False,
        },
    ]

    metrics = summarize_test_failure_modes(test_results)

    assert metrics["test/support/actionable_rows"] == pytest.approx(3.0)
    assert metrics["test/support/extractable_rows"] == pytest.approx(2.0)
    assert metrics["test/support/keypad_rows"] == pytest.approx(1.0)

    assert metrics["test/failure/wrong_tool_rate"] == pytest.approx(4 / 6)
    assert metrics["test/failure/wrong_args_rate"] == pytest.approx(1 / 6)
    assert metrics["test/failure/wait_on_actionable_rate"] == pytest.approx(1 / 3)
    assert metrics["test/failure/speak_on_keypad_rate"] == pytest.approx(1.0)
    assert metrics["test/failure/say_no_misuse_rate"] == pytest.approx(1 / 6)
    assert metrics["test/failure/missed_extract_rate"] == pytest.approx(0.5)
    assert metrics["test/failure/enter_digit_arg_mismatch_rate"] == pytest.approx(1 / 3)

    assert metrics["test/action_rate/model/wait"] == pytest.approx(2 / 6)
    assert metrics["test/action_rate/model/hangup_and_extract"] == pytest.approx(1 / 6)
    assert metrics["test/action_rate/model/enter_digit"] == pytest.approx(1 / 6)
    assert metrics["test/support/model_action_rows"] == pytest.approx(6.0)
