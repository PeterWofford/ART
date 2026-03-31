"""Core data models for conversations, messages, and validation issues."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Issue(BaseModel):
    """A single validation issue found in the dataset."""

    check_id: str
    severity: Severity
    conversation_index: int  # line number (0-based) or array index
    conversation_id: str | None = None
    message_index: int | None = None
    description: str
    context: dict[str, Any] | None = None


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    arguments: Any | None = None

    model_config = {"extra": "allow"}


class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolFunction

    model_config = {"extra": "allow"}


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction

    model_config = {"extra": "allow"}


class Message(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    timestamp: str | None = None

    model_config = {"extra": "allow"}


class Conversation(BaseModel):
    """A single conversation, normalized from various input formats."""

    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    conversation_id: str | None = None

    model_config = {"extra": "allow"}
