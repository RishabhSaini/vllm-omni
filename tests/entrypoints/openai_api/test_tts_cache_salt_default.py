# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the default TTS KV prefix-cache salt."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import _ensure_cache_salt
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_adapter():
    return MagicMock(spec=ARTTSAdapter)


def _request(**overrides):
    fields = {"input": "Hello from the test.", "voice": "vivian"}
    fields.update(overrides)
    return OpenAICreateSpeechRequest(**fields)


def test_default_salt_applies_to_unsalted_ar_prompt() -> None:
    adapter = _ar_adapter()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert isinstance(prompt.get("cache_salt"), str) and prompt["cache_salt"]


def test_default_salt_stable_and_sensitive() -> None:
    adapter = _ar_adapter()
    first: dict = {}
    _ensure_cache_salt(adapter, _request(), first, {})
    repeat: dict = {}
    _ensure_cache_salt(adapter, _request(), repeat, {})
    assert repeat["cache_salt"] == first["cache_salt"]

    other_voice: dict = {}
    _ensure_cache_salt(adapter, _request(voice="marcus"), other_voice, {})
    assert other_voice["cache_salt"] != first["cache_salt"]

    other_text: dict = {}
    _ensure_cache_salt(adapter, _request(input="Something else entirely."), other_text, {})
    assert other_text["cache_salt"] != first["cache_salt"]


def test_explicit_salt_wins() -> None:
    adapter = _ar_adapter()
    prompt = {"cache_salt": "adapter-provided"}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert prompt["cache_salt"] == "adapter-provided"


def test_non_ar_backend_skipped() -> None:
    adapter = object()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert "cache_salt" not in prompt
