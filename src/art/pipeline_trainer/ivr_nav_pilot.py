from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class IVRNavPilotConfig:
    """Minimal config surface for a PipelineTrainer-backed IVR nav pilot."""

    rollout_temperature: float = 1.0
    eval_temperature: float = 0.0
    n_holdout_rows: int = 300
    n_test_rows: int = 300
    shuffle_seed: int = 42

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "IVRNavPilotConfig":
        return cls(
            rollout_temperature=float(env.get("ROLLOUT_TEMPERATURE", "1.0")),
            eval_temperature=float(env.get("EVAL_TEMPERATURE", "0.0")),
            n_holdout_rows=int(env.get("N_HOLDOUT_ROWS", "300")),
            n_test_rows=int(env.get("N_TEST_ROWS", "300")),
            shuffle_seed=int(env.get("SHUFFLE_SEED", "42")),
        )


def row_session_id(row: Mapping[str, Any], row_index: int) -> str:
    sync_id = row.get("sync_id")
    if isinstance(sync_id, str) and sync_id:
        return sync_id

    output = row.get("output")
    if isinstance(output, Mapping):
        output_id = output.get("id")
        if isinstance(output_id, str) and output_id:
            return output_id

    return f"row-{row_index}"


def split_train_holdout_by_session(
    rows: Sequence[dict[str, Any]],
    *,
    n_holdout_rows: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows by session so one call cannot leak across train and holdout."""

    if n_holdout_rows <= 0 or not rows:
        return [], list(rows)

    rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_order: list[str] = []
    for idx, row in enumerate(rows):
        session_id = row_session_id(row, idx)
        if session_id not in rows_by_session:
            session_order.append(session_id)
        rows_by_session[session_id].append(row)

    rng = random.Random(shuffle_seed)
    rng.shuffle(session_order)

    holdout_rows: list[dict[str, Any]] = []
    holdout_sessions: set[str] = set()
    for session_id in session_order:
        holdout_rows.extend(rows_by_session[session_id])
        holdout_sessions.add(session_id)
        if len(holdout_rows) >= n_holdout_rows:
            break

    train_rows = [
        row
        for idx, row in enumerate(rows)
        if row_session_id(row, idx) not in holdout_sessions
    ]
    return holdout_rows, train_rows


def select_test_rows(
    rows: Sequence[dict[str, Any]],
    *,
    n_test_rows: int,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    """Select eval rows, preferring split == TEST when that field exists."""

    if not rows or n_test_rows <= 0:
        return []

    candidate_rows = list(rows)
    if any("split" in row for row in candidate_rows):
        filtered_rows = [row for row in candidate_rows if row.get("split") == "TEST"]
        if filtered_rows:
            candidate_rows = filtered_rows

    rng = random.Random(shuffle_seed)
    rng.shuffle(candidate_rows)
    return candidate_rows[:n_test_rows]
