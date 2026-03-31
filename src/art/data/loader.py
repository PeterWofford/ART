"""Load and normalize JSONL/JSON conversation data from various formats.

Handles:
- Bare array format: [{role, content}, ...] (e.g. SquadStack)
- Object format: {messages: [...], tools: [...]} (e.g. Bonnie)
- Directory of JSONL/JSON files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Conversation, Issue, Message, Severity, ToolDefinition


def _normalize_line(
    raw: Any, line_index: int
) -> tuple[Conversation | None, Issue | None]:
    """Normalize a single parsed JSON value into a Conversation."""
    try:
        if isinstance(raw, list):
            messages = [Message.model_validate(m) for m in raw]
            return Conversation(messages=messages), None

        if isinstance(raw, dict):
            if "messages" in raw:
                messages = [Message.model_validate(m) for m in raw["messages"]]
                tools = None
                if "tools" in raw and raw["tools"]:
                    tools = [ToolDefinition.model_validate(t) for t in raw["tools"]]
                conv_id = raw.get("conversation_id") or raw.get("id")
                return Conversation(
                    messages=messages,
                    tools=tools,
                    conversation_id=str(conv_id) if conv_id is not None else None,
                ), None
            else:
                return None, Issue(
                    check_id="loader.missing_messages_key",
                    severity=Severity.ERROR,
                    conversation_index=line_index,
                    description=(
                        f"Line {line_index}: JSON object has no 'messages' key. "
                        f"Found keys: {list(raw.keys())[:10]}. "
                        "Expected either a messages array or an object with a 'messages' field."
                    ),
                )

        return None, Issue(
            check_id="loader.unexpected_type",
            severity=Severity.ERROR,
            conversation_index=line_index,
            description=f"Line {line_index}: Expected JSON array or object, got {type(raw).__name__}.",
        )
    except Exception as e:
        return None, Issue(
            check_id="loader.parse_error",
            severity=Severity.ERROR,
            conversation_index=line_index,
            description=f"Line {line_index}: Failed to parse conversation: {e}",
        )


def load_jsonl(path: Path) -> tuple[list[tuple[int, Conversation]], list[Issue]]:
    """Load a JSONL file, returning (indexed conversations, load-time issues)."""
    conversations: list[tuple[int, Conversation]] = []
    issues: list[Issue] = []

    with open(path) as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(
                    Issue(
                        check_id="loader.invalid_json",
                        severity=Severity.ERROR,
                        conversation_index=line_index,
                        description=f"Line {line_index}: Invalid JSON: {e}",
                    )
                )
                continue

            conv, issue = _normalize_line(raw, line_index)
            if issue:
                issues.append(issue)
            if conv:
                conversations.append((line_index, conv))

    return conversations, issues


def load_json_array(path: Path) -> tuple[list[tuple[int, Conversation]], list[Issue]]:
    """Load a JSON file containing an array of conversation objects."""
    conversations: list[tuple[int, Conversation]] = []
    issues: list[Issue] = []

    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(
                Issue(
                    check_id="loader.invalid_json",
                    severity=Severity.ERROR,
                    conversation_index=0,
                    description=f"Failed to parse {path.name}: {e}",
                )
            )
            return conversations, issues

    if not isinstance(data, list):
        issues.append(
            Issue(
                check_id="loader.unexpected_type",
                severity=Severity.ERROR,
                conversation_index=0,
                description=f"{path.name}: Expected JSON array at top level, got {type(data).__name__}.",
            )
        )
        return conversations, issues

    for i, item in enumerate(data):
        conv, issue = _normalize_line(item, i)
        if issue:
            issues.append(issue)
        if conv:
            conversations.append((i, conv))

    return conversations, issues


def load_path(path: Path) -> tuple[list[tuple[int, Conversation]], list[Issue]]:
    """Load conversations from a file or directory."""
    if path.is_dir():
        all_convs: list[tuple[int, Conversation]] = []
        all_issues: list[Issue] = []
        for child in sorted(path.iterdir()):
            if child.suffix in (".jsonl", ".json"):
                convs, issues = load_path(child)
                all_convs.extend(convs)
                all_issues.extend(issues)
        return all_convs, all_issues

    if path.suffix == ".jsonl":
        return load_jsonl(path)
    elif path.suffix == ".json":
        return load_json_array(path)
    else:
        return [], [
            Issue(
                check_id="loader.unsupported_format",
                severity=Severity.ERROR,
                conversation_index=0,
                description=f"Unsupported file format: {path.suffix}. Expected .jsonl or .json.",
            )
        ]
