"""Conversation completeness checks."""

from __future__ import annotations

from ..models import Conversation, Issue, Severity
from . import register


@register(
    "completeness.min_turns",
    "Minimum turns",
    "Conversations have at least a system prompt + one exchange",
)
def check_min_turns(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        non_system = [m for m in conv.messages if m.role != "system"]
        if len(non_system) < 2:
            issues.append(
                Issue(
                    check_id="completeness.min_turns",
                    severity=Severity.WARNING,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description=(
                        f"Conversation has only {len(non_system)} non-system message(s). "
                        "Minimal conversations provide little training signal."
                    ),
                )
            )
    return issues


@register(
    "completeness.has_assistant_message",
    "Has assistant message",
    "At least one assistant message exists",
)
def check_has_assistant(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        if not any(m.role == "assistant" for m in conv.messages):
            issues.append(
                Issue(
                    check_id="completeness.has_assistant_message",
                    severity=Severity.ERROR,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description="Conversation has no assistant messages — nothing for the model to learn from.",
                )
            )
    return issues


@register(
    "completeness.pending_tool_calls",
    "Pending tool calls",
    "Conversation doesn't end with unresolved non-terminal tool calls",
)
def check_pending_tool_calls(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    terminal = {"hangup_call", "end_phone_call", "end_call", "transfer_phone_call"}
    issues = []
    for idx, conv in conversations:
        if not conv.messages:
            continue

        # Find last assistant message with tool_calls
        last_tc_msg = None
        for msg in reversed(conv.messages):
            if msg.role == "assistant" and msg.tool_calls:
                last_tc_msg = msg
                break

        if not last_tc_msg or not last_tc_msg.tool_calls:
            continue

        # Check if all tool calls are terminal
        non_terminal = [
            tc.function.name
            for tc in last_tc_msg.tool_calls
            if tc.function.name.lower() not in terminal
        ]

        if non_terminal:
            # Check if there are tool results after
            found_result = False
            past_last_tc = False
            for msg in conv.messages:
                if msg is last_tc_msg:
                    past_last_tc = True
                    continue
                if past_last_tc and msg.role == "tool":
                    found_result = True
                    break

            if not found_result:
                issues.append(
                    Issue(
                        check_id="completeness.pending_tool_calls",
                        severity=Severity.WARNING,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        description=(
                            f"Conversation ends with unresolved tool call(s): "
                            f"{', '.join(non_terminal)}. The conversation may have been truncated."
                        ),
                    )
                )
    return issues


@register(
    "completeness.empty_conversations",
    "Empty conversations",
    "Conversations with no real content (only system messages or empty)",
)
def check_empty(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        if not conv.messages:
            issues.append(
                Issue(
                    check_id="completeness.empty_conversations",
                    severity=Severity.ERROR,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description="Conversation has no messages at all.",
                )
            )
            continue

        roles = {m.role for m in conv.messages}
        if roles <= {"system"}:
            issues.append(
                Issue(
                    check_id="completeness.empty_conversations",
                    severity=Severity.ERROR,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description="Conversation contains only system messages — no actual conversation.",
                )
            )
    return issues
