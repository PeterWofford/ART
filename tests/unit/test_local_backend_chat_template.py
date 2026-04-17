from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from art import TrainableModel
from art.local import LocalBackend


class _FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = "stock-template"


def test_load_training_tokenizer_applies_inline_chat_template(tmp_path: Path) -> None:
    backend = LocalBackend(path=str(tmp_path))
    inline = "{{- 'inline-template' }}"
    model = TrainableModel(
        name="chat-template-inline",
        project="chat-template-tests",
        base_model="fake-base-model",
        base_path=str(tmp_path),
        _internal_config={"chat_template": inline},
    )

    with patch(
        "art.local.backend.AutoTokenizer.from_pretrained",
        return_value=_FakeTokenizer(),
    ):
        tokenizer = backend._load_training_tokenizer(model)

    assert tokenizer.chat_template == inline
    # Cache should be warm for future calls.
    assert backend._tokenizers["fake-base-model"] is tokenizer


def test_load_training_tokenizer_reads_template_file(tmp_path: Path) -> None:
    backend = LocalBackend(path=str(tmp_path))
    template_path = tmp_path / "qwen3_5_multi_system.jinja"
    template_body = "{{- 'file-template-contents' }}"
    template_path.write_text(template_body)

    model = TrainableModel(
        name="chat-template-file",
        project="chat-template-tests",
        base_model="fake-base-model-file",
        base_path=str(tmp_path),
        _internal_config={"chat_template": str(template_path)},
    )

    with patch(
        "art.local.backend.AutoTokenizer.from_pretrained",
        return_value=_FakeTokenizer(),
    ):
        tokenizer = backend._load_training_tokenizer(model)

    assert tokenizer.chat_template == template_body


def test_load_training_tokenizer_leaves_default_when_unset(tmp_path: Path) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name="chat-template-default",
        project="chat-template-tests",
        base_model="fake-base-model-default",
        base_path=str(tmp_path),
    )

    with patch(
        "art.local.backend.AutoTokenizer.from_pretrained",
        return_value=_FakeTokenizer(),
    ):
        tokenizer = backend._load_training_tokenizer(model)

    # No override requested, so the tokenizer keeps whatever AutoTokenizer loaded.
    assert tokenizer.chat_template == "stock-template"
