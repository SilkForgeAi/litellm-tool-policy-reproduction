"""
Minimal executable reproduction for LiteLLM ToolPolicyGuardrail:
1. Stage 2 trust-chain evaluation (`_resolve_tool_names_from_messages` and message inspection)
   only inspects OpenAI Chat Completions `role: "tool"` / `tool_calls` shapes, skipping
   Anthropic `/v1/messages` `tool_use` / `tool_result` blocks.
2. Stage 1 request tool extraction skips `extract_request_tool_names(route, request_data)`
   whenever `function_tool_names` is non-empty (`if route and not function_tool_names:`).

Pinned upstream target:
- Repository: BerriAI/litellm
- Commit: e63fc85b7b6db8ea1bc0f444f3b997de71898b66
- File: litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.tool_policy.tool_policy_guardrail import (
    ToolPolicyGuardrail,
)


def _registry_mock(input_policies: dict, output_policies: dict | None = None) -> MagicMock:
    """Return a mock ToolPolicyRegistry matching upstream test_tool_policy_guardrail.py."""
    reg = MagicMock()
    reg.is_initialized.return_value = True
    reg.get_effective_policies.return_value = input_policies
    out_map = output_policies or {}
    reg.get_output_policy.side_effect = lambda name: out_map.get(name, "untrusted")
    return reg


def _tool_request_inputs(tool_names: list[str]) -> dict[str, Any]:
    return {
        "tools": [
            {"type": "function", "function": {"name": name, "description": ""}}
            for name in tool_names
        ]
    }


def _tool_response_inputs(tool_names: list[str]) -> dict[str, Any]:
    return {
        "tool_calls": [
            {"type": "function", "function": {"name": name}} for name in tool_names
        ]
    }


# ---------------------------------------------------------------------------
# Passing Control Assertions (Expected upstream behavior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_openai_format_blocks_trusted_tool_after_untrusted_tool_output():
    """
    Control: In OpenAI Chat Completions format (`role: "assistant"` with `tool_calls` +
    `role: "tool"` with `tool_call_id`), `send_summary` (`input_policy="trusted"`) is
    blocked with HTTP 400 after `read_external_note` (`output_policy="untrusted"`).
    """
    guardrail = ToolPolicyGuardrail()
    input_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}
    output_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}

    request_data = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_external_note", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Note contents",
            },
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies, output_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_tool_response_inputs(["send_summary"]),
                request_data=request_data,
                input_type="response",
            )
    assert exc_info.value.status_code == 400
    assert "send_summary" in exc_info.value.detail["blocked_tools"]
    assert "read_external_note" in exc_info.value.detail["untrusted_sources"]


@pytest.mark.asyncio
async def test_control_openai_format_allows_trusted_tool_after_trusted_tool_output():
    """
    Control: When the preceding tool output has `output_policy="trusted"` (`ping`),
    calling `send_summary` (`input_policy="trusted"`) succeeds without raising HTTPException.
    """
    guardrail = ToolPolicyGuardrail()
    input_policies = {"ping": "untrusted", "send_summary": "trusted"}
    output_policies = {"ping": "trusted", "send_summary": "trusted"}

    request_data = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ping", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "pong",
            },
        ],
    }

    inputs = _tool_response_inputs(["send_summary"])
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies, output_policies),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )
    assert result is inputs


@pytest.mark.asyncio
async def test_control_request_stage_blocks_route_extracted_tool_when_function_tools_empty():
    """
    Control: On `/v1/messages`, when `inputs["tools"]` does not contain nested OpenAI
    `{"function": {"name": ...}}` dicts (so `function_tool_names` is empty),
    `extract_request_tool_names("/v1/messages", request_data)` extracts Anthropic tool
    definitions (`{"name": "archive_record", "input_schema": {...}}`) and raises HTTP 400
    for `archive_record` (`input_policy="blocked"`).
    """
    guardrail = ToolPolicyGuardrail()
    input_policies = {"archive_record": "blocked", "ping": "untrusted"}

    request_data = {
        "user_api_key_request_route": "/v1/messages",
        "tools": [
            {"name": "archive_record", "input_schema": {"type": "object"}},
            {"name": "ping", "input_schema": {"type": "object"}},
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"tools": request_data["tools"]},
                request_data=request_data,
                input_type="request",
            )
    assert exc_info.value.status_code == 400
    assert "archive_record" in exc_info.value.detail["blocked_tools"]


# ---------------------------------------------------------------------------
# Failing Reproduction Assertions (Actual vs. Expected)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repro_anthropic_messages_tool_result_blocks_trusted_tool_call():
    """
    Bug 1 (tool_policy_guardrail.py lines 92-112 and 204-221):
    In Anthropic `/v1/messages` conversations, assistant tool invocations are recorded as
    `{"type": "tool_use", "id": "...", "name": "read_external_note"}` content blocks and tool
    outputs are returned in `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}`.
    Because `_resolve_tool_names_from_messages()` only inspects `msg.get("tool_calls")` and
    `apply_guardrail()` skips any message where `msg.get("role") != "tool"`, `untrusted_sources`
    is empty (`[]`) and `send_summary` (`input_policy="trusted"`) is allowed instead of raising HTTP 400.
    """
    guardrail = ToolPolicyGuardrail()
    input_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}
    output_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}

    request_data = {
        "user_api_key_request_route": "/v1/messages",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "read_external_note",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "Note contents",
                    }
                ],
            },
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies, output_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_tool_response_inputs(["send_summary"]),
                request_data=request_data,
                input_type="response",
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_repro_request_stage_blocks_route_tool_when_mixed_with_openai_function_tool():
    """
    Bug 2 (tool_policy_guardrail.py lines 150-154):
    On `input_type == "request"`, `extract_request_tool_names(route, request_data)` is gated by
    `if route and not function_tool_names:`. When a request on `/v1/responses` or `/v1/messages`
    includes both an OpenAI-shaped function tool (`{"type": "function", "function": {"name": "ping"}}`)
    and a flat/route-specific tool (`{"type": "function", "name": "archive_record"}`),
    `function_tool_names` is non-empty (`["ping"]`), so `extract_request_tool_names` is skipped
    and `archive_record` (`input_policy="blocked"`) is not blocked.
    """
    guardrail = ToolPolicyGuardrail()
    input_policies = {"ping": "untrusted", "archive_record": "blocked"}

    mixed_tools = [
        {"type": "function", "function": {"name": "ping", "description": ""}},
        {"type": "function", "name": "archive_record"},
    ]
    request_data = {
        "user_api_key_request_route": "/v1/responses",
        "tools": mixed_tools,
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"tools": mixed_tools},
                request_data=request_data,
                input_type="request",
            )
    assert exc_info.value.status_code == 400
    assert "archive_record" in exc_info.value.detail["blocked_tools"]
