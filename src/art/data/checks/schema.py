"""Schema and structural validation checks."""

from __future__ import annotations

import re

from ..models import Conversation, Issue, Severity
from . import register

VALID_ROLES = {"system", "user", "assistant", "tool"}


@register("schema.valid_roles", "Valid roles", "All messages have a recognized role")
def check_valid_roles(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if msg.role not in VALID_ROLES:
                issues.append(
                    Issue(
                        check_id="schema.valid_roles",
                        severity=Severity.ERROR,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        message_index=j,
                        description=(
                            f"Message {j} has invalid role '{msg.role}'. "
                            f"Expected one of: {', '.join(sorted(VALID_ROLES))}."
                        ),
                    )
                )
    return issues


@register(
    "schema.starts_with_system",
    "Starts with system",
    "First message is a system prompt",
)
def check_starts_with_system(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        if conv.messages and conv.messages[0].role != "system":
            issues.append(
                Issue(
                    check_id="schema.starts_with_system",
                    severity=Severity.WARNING,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description=(
                        f"Conversation starts with role '{conv.messages[0].role}' instead of 'system'. "
                        "Most training data should begin with a system prompt."
                    ),
                )
            )
    return issues


@register(
    "schema.tool_message_has_id",
    "Tool messages have ID",
    "Messages with role 'tool' have a tool_call_id",
)
def check_tool_message_has_id(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if msg.role == "tool" and not msg.tool_call_id:
                issues.append(
                    Issue(
                        check_id="schema.tool_message_has_id",
                        severity=Severity.ERROR,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        message_index=j,
                        description=(
                            f"Message {j} has role 'tool' but no tool_call_id. "
                            "Tool result messages must reference the tool call they respond to."
                        ),
                    )
                )
    return issues


@register(
    "schema.mid_conversation_system",
    "Mid-conversation system messages",
    "Flags system messages after conversation has started (informational)",
)
def check_mid_conversation_system(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        saw_non_system = False
        mid_system_count = 0
        for msg in conv.messages:
            if msg.role != "system":
                saw_non_system = True
            elif saw_non_system:
                mid_system_count += 1

        if mid_system_count > 0:
            issues.append(
                Issue(
                    check_id="schema.mid_conversation_system",
                    severity=Severity.INFO,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description=(
                        f"Conversation has {mid_system_count} system message(s) injected mid-conversation. "
                        "This is normal for voice AI (STT confidence, language switching) but verify "
                        "the model sees these at inference time too."
                    ),
                )
            )
    return issues


@register(
    "schema.unrendered_templates",
    "Unrendered templates",
    "Detect template variables that weren't interpolated (e.g. Jinja, mustache)",
)
def check_unrendered_templates(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    # Patterns for common template syntaxes
    patterns = [
        (re.compile(r"\{%.*?%\}"), "Jinja block"),
        (re.compile(r"\{\{.*?\}\}"), "Jinja/mustache variable"),
        (re.compile(r"<system-prompt>"), "placeholder tag"),
    ]
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if not msg.content:
                continue
            for pattern, label in patterns:
                matches = pattern.findall(msg.content)
                if matches:
                    issues.append(
                        Issue(
                            check_id="schema.unrendered_templates",
                            severity=Severity.ERROR,
                            conversation_index=idx,
                            conversation_id=conv.conversation_id,
                            message_index=j,
                            description=(
                                f"Message {j} (role={msg.role}) contains unrendered {label}: "
                                f"'{matches[0][:80]}'. System messages should be fully interpolated "
                                "with real values, not template variables."
                            ),
                            context={"pattern": label, "matches": matches[:3]},
                        )
                    )
                    break  # one issue per message is enough
    return issues


@register(
    "schema.double_encoded_content",
    "Double-encoded content",
    "Detect system message content that is JSON-stringified (double-encoded)",
)
def check_double_encoded_content(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        if not conv.messages:
            continue
        first = conv.messages[0]
        if first.role == "system" and first.content:
            content = first.content.strip()
            if (
                content.startswith("[{") or content.startswith('["')
            ) and content.endswith("]"):
                issues.append(
                    Issue(
                        check_id="schema.double_encoded_content",
                        severity=Severity.WARNING,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        message_index=0,
                        description=(
                            "System message content appears to be a JSON-encoded string "
                            "(starts with '[{' or '[\"'). This may cause the model to learn "
                            "JSON-escaped output. Consider unwrapping to plain text."
                        ),
                    )
                )
            elif content.startswith("{'") and content.endswith("}"):
                issues.append(
                    Issue(
                        check_id="schema.double_encoded_content",
                        severity=Severity.WARNING,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        message_index=0,
                        description=(
                            "System message content appears to be a Python repr/dict string. "
                            "Consider converting to plain text."
                        ),
                    )
                )
    return issues


@register(
    "schema.duplicate_conversation_ids",
    "Duplicate conversation IDs",
    "Check for duplicate conversation_id values across the dataset",
)
def check_duplicate_ids(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    seen: dict[str, int] = {}
    issues = []
    for idx, conv in conversations:
        cid = conv.conversation_id
        if cid is None:
            continue
        if cid in seen:
            issues.append(
                Issue(
                    check_id="schema.duplicate_conversation_ids",
                    severity=Severity.WARNING,
                    conversation_index=idx,
                    conversation_id=cid,
                    description=(
                        f"Duplicate conversation_id '{cid}' — first seen at line {seen[cid]}. "
                        "Duplicate conversations may skew training."
                    ),
                )
            )
        else:
            seen[cid] = idx
    return issues
