"""
Minimal executable reproduction for LiteLLM ToolPolicyGuardrail:
1. Stage 2 trust-chain evaluation (`_resolve_tool_names_from_messages` and message inspection)
   only inspects OpenAI Chat Completions `role: "tool"` / `tool_calls` shapes, skipping
   Anthropic `/v1/messages` `tool_use` / `tool_result` blocks and `/v1/responses`
   `function_call` / `function_call_output` items when invoked via `AnthropicMessagesHandler`
   and `OpenAIResponsesHandler`.
2. Stage 1 request tool extraction skips `extract_request_tool_names(route, request_data)`
   whenever `function_tool_names` is non-empty (`if route and not function_tool_names:`).
   In `AnthropicMessagesHandler.process_input_messages` (`handler.py:639-647`), provider-native
   tools (`{"type": "bash_20250124", "name": "bash"}`) are excluded from `inputs["tools"]`
   via `if not is_provider_native_tool_dict(tool)`. When sent alone, `function_tool_names` is
   empty so `extract_request_tool_names` blocks `bash`; when sent alongside any regular tool
   (`ping`), `function_tool_names` is `["ping"]`, `extract_request_tool_names` is skipped,
   and blocked `bash` is allowed through.

Pinned upstream target:
- Repository: BerriAI/litellm
- Commit: e63fc85b7b6db8ea1bc0f444f3b997de71898b66
- File: litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.llms.anthropic.chat.guardrail_translation.handler import (
    AnthropicMessagesHandler,
)
from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
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
async def test_control_anthropic_handler_blocks_solo_provider_native_bash_tool():
    """
    Control: Through `AnthropicMessagesHandler().process_input_messages()`, when `/v1/messages`
    contains only a provider-native tool `{"type": "bash_20250124", "name": "bash"}` (`blocked`),
    `tools_to_check` is empty (`function_tool_names == []`), so `ToolPolicyGuardrail` calls
    `extract_request_tool_names("/v1/messages", request_data)` and raises HTTP 400 for `bash`.
    """
    guardrail = ToolPolicyGuardrail()
    handler = AnthropicMessagesHandler()
    input_policies = {"bash": "blocked", "ping": "untrusted"}

    request_data = {
        "model": "claude-sonnet-4-20250514",
        "user_api_key_request_route": "/v1/messages",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "bash_20250124", "name": "bash"}],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_input_messages(request_data, guardrail)
    assert exc_info.value.status_code == 400
    assert "bash" in exc_info.value.detail["blocked_tools"]


# ---------------------------------------------------------------------------
# Failing Reproduction Assertions (Actual vs. Expected)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repro_anthropic_handler_output_response_blocks_trusted_tool_after_untrusted_tool_result():
    """
    Bug 1a (tool_policy_guardrail.py lines 92-112 and 204-221):
    End-to-end through `AnthropicMessagesHandler().process_output_response()`:
    The handler extracts the response `tool_use` block (`send_summary`, `input_policy="trusted"`)
    into `inputs["tool_calls"]` and passes the `/v1/messages` `request_data` to `apply_guardrail()`.
    Because `_resolve_tool_names_from_messages()` only inspects `msg.get("tool_calls")` and
    `apply_guardrail()` skips any message where `msg.get("role") != "tool"`, the Anthropic
    `tool_use` and `role: "user"` `tool_result` blocks for `read_external_note` (`output_policy="untrusted"`)
    are ignored and `send_summary` is allowed instead of raising HTTP 400.
    """
    guardrail = ToolPolicyGuardrail()
    handler = AnthropicMessagesHandler()
    input_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}
    output_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}

    request_data = {
        "model": "claude-sonnet-4-20250514",
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
    response: Any = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_02",
                "name": "send_summary",
                "input": {"text": "summary"},
            }
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies, output_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(
                response, guardrail, request_data=request_data
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_repro_responses_handler_output_response_blocks_trusted_tool_after_untrusted_output():
    """
    Bug 1b (tool_policy_guardrail.py lines 92-112 and 204-221):
    End-to-end through `OpenAIResponsesHandler().process_output_response()`:
    On `/v1/responses`, prior tool calls and outputs are carried in `request_data["input"]`
    as `{"type": "function_call", ...}` and `{"type": "function_call_output", ...}` items
    (not `request_data["messages"]`). `ToolPolicyGuardrail` inspects only `request_data.get("messages")`,
    so `untrusted_sources` is `[]` and `send_summary` (`input_policy="trusted"`) is allowed.
    """
    guardrail = ToolPolicyGuardrail()
    handler = OpenAIResponsesHandler()
    input_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}
    output_policies = {"read_external_note": "untrusted", "send_summary": "trusted"}

    request_data = {
        "model": "gpt-4o",
        "user_api_key_request_route": "/v1/responses",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_external_note",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Note contents",
            },
        ],
    }
    response: Any = {
        "id": "resp_01",
        "model": "gpt-4o",
        "output": [
            {
                "type": "function_call",
                "id": "fc_02",
                "call_id": "call_2",
                "name": "send_summary",
                "arguments": "{}",
            }
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies, output_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_output_response(
                response, guardrail, request_data=request_data
            )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_repro_anthropic_handler_blocks_provider_native_bash_tool_when_mixed_with_custom_tool():
    """
    Bug 2 (tool_policy_guardrail.py lines 142-154 + anthropic handler.py lines 639-647):
    End-to-end through `AnthropicMessagesHandler().process_input_messages()`:
    `AnthropicMessagesHandler.process_input_messages()` excludes provider-native tools
    (`is_provider_native_tool_dict(tool)`, such as `{"type": "bash_20250124", "name": "bash"}`)
    from `inputs["tools"]` (`handler.py:645`). When the request also contains any standard custom
    tool (`{"name": "ping", "input_schema": {"type": "object"}}`), `inputs["tools"]` contains
    `ping`, so `function_tool_names` in `ToolPolicyGuardrail.apply_guardrail()` is `["ping"]`
    (non-empty). Because `extract_request_tool_names(route, request_data)` is gated by
    `if route and not function_tool_names:` (`tool_policy_guardrail.py:152`), `bash` is never
    extracted or checked against `ToolPolicyRegistry` and remains in `request_data["tools"]`.
    """
    guardrail = ToolPolicyGuardrail()
    handler = AnthropicMessagesHandler()
    input_policies = {"bash": "blocked", "ping": "untrusted"}

    request_data = {
        "model": "claude-sonnet-4-20250514",
        "user_api_key_request_route": "/v1/messages",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {"name": "ping", "input_schema": {"type": "object"}},
            {"type": "bash_20250124", "name": "bash"},
        ],
    }

    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(input_policies),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handler.process_input_messages(request_data, guardrail)
    assert exc_info.value.status_code == 400
    assert "bash" in exc_info.value.detail["blocked_tools"]
