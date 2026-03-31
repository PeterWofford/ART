"""Tool definition, tool call, and tool result integrity checks."""

from __future__ import annotations

from collections import Counter
import json
import re

from ..models import Conversation, Issue, Severity
from . import register

# Tools that are terminal (no result expected because the conversation ends)
TERMINAL_TOOLS = {"hangup_call", "end_phone_call", "end_call", "transfer_phone_call"}


def _get_defined_tool_names(conv: Conversation) -> set[str]:
    """Get tool names defined in the conversation's tools array."""
    if not conv.tools:
        return set()
    return {t.function.name for t in conv.tools}


def _conversation_has_tool_calls(conv: Conversation) -> bool:
    return any(msg.tool_calls for msg in conv.messages)


def _parse_arguments(args: any) -> dict | None:
    """Parse tool call arguments whether they're a string or already a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


@register(
    "tool.definitions_present",
    "Tool definitions present",
    "If tool calls exist, verify that tool definitions are provided",
)
def check_definitions_present(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    convos_with_calls = sum(
        1 for _, c in conversations if _conversation_has_tool_calls(c)
    )
    convos_with_tools = sum(1 for _, c in conversations if c.tools)

    issues = []
    if convos_with_calls > 0 and convos_with_tools == 0:
        issues.append(
            Issue(
                check_id="tool.definitions_present",
                severity=Severity.WARNING,
                conversation_index=-1,
                description=(
                    f"Dataset has {convos_with_calls} conversations with tool calls but NO tool "
                    "definitions in any conversation. Tool definitions may be provided separately, "
                    "but consider embedding them in the data or supplying via --tools flag."
                ),
                context={
                    "conversations_with_calls": convos_with_calls,
                    "conversations_with_definitions": convos_with_tools,
                },
            )
        )
    return issues


@register(
    "tool.no_tool_calls_in_dataset",
    "No tool calls in dataset",
    "Flag if dataset has tool definitions but zero tool calls",
)
def check_no_tool_calls(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    convos_with_calls = sum(
        1 for _, c in conversations if _conversation_has_tool_calls(c)
    )
    convos_with_tools = sum(1 for _, c in conversations if c.tools)

    issues = []
    if convos_with_tools > 0 and convos_with_calls == 0:
        tool_names = set()
        for _, c in conversations:
            tool_names.update(_get_defined_tool_names(c))
        issues.append(
            Issue(
                check_id="tool.no_tool_calls_in_dataset",
                severity=Severity.ERROR,
                conversation_index=-1,
                description=(
                    f"Dataset defines tools ({', '.join(sorted(tool_names))}) but no conversation "
                    "contains any tool calls. The data may have been truncated or tool calls "
                    "were lost during export."
                ),
            )
        )

    # Also flag if NO tool definitions AND NO tool calls but system prompts
    # reference tool-like actions (trigger hangup_call, invoke X, etc.)
    if convos_with_tools == 0 and convos_with_calls == 0:
        # Look for function-name patterns: snake_case with common suffixes,
        # or keywords like "trigger", "invoke", "call function"
        tool_mention_convos = 0
        detected_names = set()
        for _, c in conversations:
            for msg in c.messages:
                if msg.role == "system" and msg.content:
                    content_lower = msg.content.lower()
                    # Direct keyword check
                    has_mention = "tool" in content_lower
                    # Function-call-like patterns in system prompts
                    if not has_mention:
                        fn_matches = re.findall(
                            r'(?:trigger|invoke|call|use)\s+[`"]?(\w+_\w+)[`"]?',
                            content_lower,
                        )
                        if fn_matches:
                            has_mention = True
                            detected_names.update(fn_matches)
                    if has_mention:
                        tool_mention_convos += 1
                        break

        if tool_mention_convos > len(conversations) * 0.3:
            name_hint = ""
            if detected_names:
                name_hint = f" Detected function references: {', '.join(sorted(detected_names)[:5])}."
            issues.append(
                Issue(
                    check_id="tool.no_tool_calls_in_dataset",
                    severity=Severity.ERROR,
                    conversation_index=-1,
                    description=(
                        f"No tool calls or definitions in dataset, but "
                        f"{tool_mention_convos}/{len(conversations)} system prompts reference "
                        f"tool-like functions.{name_hint} "
                        "Tool calls were likely lost during data export. Re-export to include "
                        "assistant tool_calls from the API responses."
                    ),
                )
            )

    return issues


@register(
    "tool.undefined_tool_reference",
    "Undefined tool references",
    "Tool calls reference tools not in the definitions array",
)
def check_undefined_references(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        defined = _get_defined_tool_names(conv)
        if not defined:
            continue  # Can't check if no definitions

        for j, msg in enumerate(conv.messages):
            if not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function.name not in defined:
                    issues.append(
                        Issue(
                            check_id="tool.undefined_tool_reference",
                            severity=Severity.ERROR,
                            conversation_index=idx,
                            conversation_id=conv.conversation_id,
                            message_index=j,
                            description=(
                                f"Tool call '{tc.function.name}' is not defined in the tools array. "
                                f"Defined tools: {', '.join(sorted(defined))}. "
                                "Add this tool to the definitions or remove the call."
                            ),
                            context={
                                "called": tc.function.name,
                                "defined": sorted(defined),
                            },
                        )
                    )
    return issues


@register(
    "tool.call_result_pairing",
    "Tool call/result pairing",
    "Every tool call should have a matching tool result (except terminal tools)",
)
def check_call_result_pairing(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        call_ids: dict[str, str] = {}  # id -> tool name
        result_ids: set[str] = set()

        for msg in conv.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id:
                        call_ids[tc.id] = tc.function.name
            if msg.role == "tool" and msg.tool_call_id:
                result_ids.add(msg.tool_call_id)

        # Check for calls without results (skip terminal tools)
        for call_id, tool_name in call_ids.items():
            if call_id not in result_ids and tool_name.lower() not in TERMINAL_TOOLS:
                issues.append(
                    Issue(
                        check_id="tool.call_result_pairing",
                        severity=Severity.ERROR,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        description=(
                            f"Tool call '{tool_name}' (id={call_id[:20]}...) has no matching "
                            "tool result message. Add a tool message with the corresponding "
                            "tool_call_id, or mark this as a terminal tool."
                        ),
                        context={"tool_name": tool_name, "call_id": call_id},
                    )
                )

        # Check for orphaned results
        for result_id in result_ids:
            if result_id not in call_ids:
                issues.append(
                    Issue(
                        check_id="tool.call_result_pairing",
                        severity=Severity.ERROR,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        description=(
                            f"Tool result with tool_call_id='{result_id[:20]}...' has no matching "
                            "tool call. This orphaned result may confuse the model during training."
                        ),
                        context={"orphaned_result_id": result_id},
                    )
                )
    return issues


@register(
    "tool.arguments_valid_json",
    "Arguments are valid",
    "Tool call arguments can be parsed (whether string or object)",
)
def check_arguments_valid(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if args is None:
                    continue
                if isinstance(args, dict):
                    continue  # Already parsed, valid
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        issues.append(
                            Issue(
                                check_id="tool.arguments_valid_json",
                                severity=Severity.ERROR,
                                conversation_index=idx,
                                conversation_id=conv.conversation_id,
                                message_index=j,
                                description=(
                                    f"Tool call '{tc.function.name}' has arguments that are not "
                                    f"valid JSON: '{str(args)[:100]}...'"
                                ),
                            )
                        )
    return issues


@register(
    "tool.repeated_tool_call_ids",
    "Repeated tool call IDs",
    "Detect the vLLM bug where tool_call IDs contain repeated substrings",
)
def check_repeated_ids(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                tid = tc.id
                if not tid or len(tid) < 40:
                    continue
                # Check for repeated substring pattern
                # e.g., "chatcmpl-tool-abc123chatcmpl-tool-abc123chatcmpl-tool-abc123"
                for chunk_len in range(15, len(tid) // 2 + 1):
                    chunk = tid[:chunk_len]
                    if (
                        tid == chunk * (len(tid) // chunk_len)
                        and len(tid) // chunk_len >= 3
                    ):
                        issues.append(
                            Issue(
                                check_id="tool.repeated_tool_call_ids",
                                severity=Severity.ERROR,
                                conversation_index=idx,
                                conversation_id=conv.conversation_id,
                                message_index=j,
                                description=(
                                    f"Tool call ID is a repeated pattern ('{chunk[:30]}...' "
                                    f"repeated {len(tid) // chunk_len}x). This is a known vLLM bug. "
                                    "These IDs may cause issues with tool result matching."
                                ),
                            )
                        )
                        break
    return issues


@register(
    "tool.duplicate_consecutive_calls",
    "Duplicate consecutive calls",
    "Same tool called with identical arguments back-to-back",
)
def check_duplicate_consecutive(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        prev_calls: list[tuple[str, str]] = []  # (name, args_normalized)
        for j, msg in enumerate(conv.messages):
            if not msg.tool_calls:
                prev_calls = []
                continue
            current_calls = []
            for tc in msg.tool_calls:
                args = _parse_arguments(tc.function.arguments)
                normalized = json.dumps(args, sort_keys=True) if args else ""
                current_calls.append((tc.function.name, normalized))

            for call in current_calls:
                if call in prev_calls:
                    issues.append(
                        Issue(
                            check_id="tool.duplicate_consecutive_calls",
                            severity=Severity.WARNING,
                            conversation_index=idx,
                            conversation_id=conv.conversation_id,
                            message_index=j,
                            description=(
                                f"Tool '{call[0]}' called with identical arguments in consecutive "
                                "assistant turns. This may train the model to repeat tool calls."
                            ),
                        )
                    )
                    break

            prev_calls = current_calls
    return issues


@register(
    "tool.unused_definitions",
    "Unused tool definitions",
    "Tools defined but never called across the dataset",
)
def check_unused_definitions(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    all_defined: Counter[str] = Counter()
    all_called: set[str] = set()

    for _, conv in conversations:
        for name in _get_defined_tool_names(conv):
            all_defined[name] += 1
        for msg in conv.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    all_called.add(tc.function.name)

    unused = set(all_defined.keys()) - all_called
    issues = []
    if unused:
        issues.append(
            Issue(
                check_id="tool.unused_definitions",
                severity=Severity.INFO,
                conversation_index=-1,
                description=(
                    f"Tools defined but never called in the dataset: {', '.join(sorted(unused))}. "
                    "The model won't learn to use these tools from this data."
                ),
                context={"unused_tools": sorted(unused)},
            )
        )
    return issues
