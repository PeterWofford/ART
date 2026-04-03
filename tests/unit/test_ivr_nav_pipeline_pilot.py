from art.pipeline_trainer.ivr_nav_pilot import (
    IVRNavPilotConfig,
    row_session_id,
    select_test_rows,
    split_train_holdout_by_session,
)


def test_pilot_config_defaults_to_greedy_eval_temperature() -> None:
    config = IVRNavPilotConfig.from_env({})

    assert config.rollout_temperature == 1.0
    assert config.eval_temperature == 0.0


def test_select_test_rows_prefers_test_split() -> None:
    rows = [
        {"messages": [], "tools": [], "split": "TRAIN", "id": "train-1"},
        {"messages": [], "tools": [], "split": "TEST", "id": "test-1"},
        {"messages": [], "tools": [], "split": "TEST", "id": "test-2"},
    ]

    selected = select_test_rows(rows, n_test_rows=8, shuffle_seed=7)

    assert len(selected) == 2
    assert {row["id"] for row in selected} == {"test-1", "test-2"}
    assert all(row["split"] == "TEST" for row in selected)


def test_select_test_rows_uses_full_file_when_split_missing() -> None:
    rows = [
        {"messages": [], "tools": [], "id": "a"},
        {"messages": [], "tools": [], "id": "b"},
        {"messages": [], "tools": [], "id": "c"},
    ]

    selected = select_test_rows(rows, n_test_rows=2, shuffle_seed=11)

    assert len(selected) == 2
    assert {row["id"] for row in selected}.issubset({"a", "b", "c"})


def test_split_train_holdout_by_session_keeps_sessions_together() -> None:
    rows = [
        {"sync_id": "s1", "payload": 1},
        {"sync_id": "s1", "payload": 2},
        {"sync_id": "s2", "payload": 3},
        {"sync_id": "s2", "payload": 4},
        {"sync_id": "s3", "payload": 5},
    ]

    holdout_rows, train_rows = split_train_holdout_by_session(
        rows,
        n_holdout_rows=2,
        shuffle_seed=3,
    )

    holdout_sessions = {row["sync_id"] for row in holdout_rows}
    train_sessions = {row["sync_id"] for row in train_rows}

    assert holdout_sessions.isdisjoint(train_sessions)
    assert len(holdout_rows) >= 2
    assert sorted(holdout_rows + train_rows, key=lambda row: row["payload"]) == rows


def test_row_session_id_falls_back_to_output_id_then_row_index() -> None:
    assert row_session_id({"sync_id": "sync-1"}, 0) == "sync-1"
    assert row_session_id({"output": {"id": "out-1"}}, 1) == "out-1"
    assert row_session_id({"messages": []}, 2) == "row-2"
