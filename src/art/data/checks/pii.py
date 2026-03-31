"""PII consistency and detection checks."""

from __future__ import annotations

from collections import defaultdict
import re

from ..models import Conversation, Issue, Severity
from . import register

PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-]{7,15}\d")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
REDACTION_PATTERNS = re.compile(
    r"\[redacted\]|\[REDACTED\]|\*{3,}|<redacted>|<PII>", re.IGNORECASE
)


def _extract_phones(text: str) -> set[str]:
    """Extract phone-number-like strings, normalized to digits only."""
    raw = PHONE_PATTERN.findall(text)
    return {re.sub(r"[\s\-]", "", p) for p in raw if len(re.sub(r"[\s\-]", "", p)) >= 8}


def _extract_emails(text: str) -> set[str]:
    return {m.lower() for m in EMAIL_PATTERN.findall(text)}


@register(
    "pii.phone_consistency",
    "Phone number consistency",
    "Phone numbers in system prompt match those referenced in assistant messages",
)
def check_phone_consistency(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        system_phones: set[str] = set()
        assistant_phones: set[str] = set()

        for msg in conv.messages:
            if not msg.content:
                continue
            if msg.role == "system":
                system_phones.update(_extract_phones(msg.content))
            elif msg.role == "assistant":
                assistant_phones.update(_extract_phones(msg.content))

        if not system_phones or not assistant_phones:
            continue

        # Check if assistant references phones not in system prompt
        novel = assistant_phones - system_phones
        if novel and system_phones:
            # Only flag if the numbers are substantially different (not just formatting)
            system_digits = {p[-4:] for p in system_phones}
            novel_different = {p for p in novel if p[-4:] not in system_digits}
            if novel_different:
                issues.append(
                    Issue(
                        check_id="pii.phone_consistency",
                        severity=Severity.WARNING,
                        conversation_index=idx,
                        conversation_id=conv.conversation_id,
                        description=(
                            f"Assistant references phone number(s) not in system prompt. "
                            f"System: {sorted(system_phones)[:2]}, "
                            f"Assistant: {sorted(novel_different)[:2]}. "
                            "This may indicate PII redaction inconsistency."
                        ),
                    )
                )
    return issues


@register(
    "pii.partial_redaction",
    "Partial PII redaction",
    "Mix of redacted and unredacted PII in the same conversation",
)
def check_partial_redaction(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        has_redaction = False
        has_raw_phone = False
        has_raw_email = False

        for msg in conv.messages:
            if not msg.content:
                continue
            if REDACTION_PATTERNS.search(msg.content):
                has_redaction = True
            if msg.role in ("user", "assistant"):
                if _extract_phones(msg.content):
                    has_raw_phone = True
                if _extract_emails(msg.content):
                    has_raw_email = True

        if has_redaction and (has_raw_phone or has_raw_email):
            issues.append(
                Issue(
                    check_id="pii.partial_redaction",
                    severity=Severity.WARNING,
                    conversation_index=idx,
                    conversation_id=conv.conversation_id,
                    description=(
                        "Conversation has both redacted markers (e.g. [redacted]) and unredacted "
                        "PII (phone/email) in user/assistant messages. PII redaction should be "
                        "consistent within a conversation."
                    ),
                )
            )
    return issues
