# LiteLLM `ToolPolicyGuardrail` Non-OpenAI Route & Provider-Native Tool Reproduction

Minimal executable reproduction for two tool-policy enforcement gaps in `BerriAI/litellm`'s `ToolPolicyGuardrail` (`litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py`), reproduced end-to-end through `AnthropicMessagesHandler` and `OpenAIResponsesHandler`.

## Pinned Target

- **Repository:** `https://github.com/BerriAI/litellm`
- **Commit:** `e63fc85b7b6db8ea1bc0f444f3b997de71898b66`
- **Package Version (`litellm`):** `1.93.0`
- **Affected Files:**
  - `litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py` (lines `92–112`, `141–155`, and `204–240`)
  - `litellm/llms/anthropic/chat/guardrail_translation/handler.py` (lines `639–647`, `1242–1273`)
  - `litellm/llms/openai/responses/guardrail_translation/handler.py` (lines `1108–1143`)

## Summary

`ToolPolicyGuardrail.apply_guardrail()` enforces two stages of tool policy from `ToolPolicyRegistry`:
- **Stage 1 (`lines 180–199`):** Raises `HTTPException(status_code=400)` when any tool in `tool_names` has effective `input_policy == "blocked"`.
- **Stage 2 (`lines 204–240`, `input_type == "response"`):** Raises `HTTPException(status_code=400)` when a response calls a tool with `input_policy == "trusted"` and the conversation history contains output from a tool with `output_policy == "untrusted"`.

When invoked through LiteLLM's unified guardrail translation handlers (`AnthropicMessagesHandler` and `OpenAIResponsesHandler`), two conditions prevent these policies from firing:

1. **Stage 2 trust-chain evaluation only inspects OpenAI Chat Completions `role == "tool"` messages (`tool_policy_guardrail.py:92–112` and `204–221`):**
   - `AnthropicMessagesHandler.process_output_response()` (`handler.py:1242–1273`) and `OpenAIResponsesHandler.process_output_response()` (`handler.py:1108–1143`) translate response tool invocations into `inputs["tool_calls"]`, and pass the endpoint's `request_data` to `ToolPolicyGuardrail.apply_guardrail(inputs, request_data, input_type="response")`.
   - Inside `ToolPolicyGuardrail.apply_guardrail()`, `_resolve_tool_names_from_messages()` (`lines 92–112`) builds `tc_id_to_name` exclusively from `msg.get("tool_calls")` on `role == "assistant"` messages, and the Stage 2 loop (`lines 211–220`) skips any message where `msg.get("role") != "tool"`.
   - On Anthropic `/v1/messages` (where tool calls/results in `request_data["messages"]` use `type: "tool_use"` and `role: "user"` `type: "tool_result"` content blocks) and on `/v1/responses` (where tool calls/outputs are stored in `request_data["input"]` as `function_call` / `function_call_output` items), `untrusted_sources` always resolves to `[]`, allowing `input_policy == "trusted"` tools to execute after `output_policy == "untrusted"` tool outputs.

2. **Stage 1 skips `extract_request_tool_names(route, request_data)` when `function_tool_names` is non-empty (`tool_policy_guardrail.py:142–154`), allowing blocked Anthropic provider-native tools (`bash_20250124`) when mixed with any custom tool:**
   - In `AnthropicMessagesHandler.process_input_messages()` (`handler.py:639–647`), provider-native tools (`is_provider_native_tool_dict(tool)`, such as `{"type": "bash_20250124", "name": "bash"}`) are excluded from `tools_to_check` (`inputs["tools"]`).
   - When `{"type": "bash_20250124", "name": "bash"}` (`input_policy == "blocked"`) is sent **alone** on `/v1/messages`, `inputs["tools"]` is empty (`function_tool_names == []`), so `ToolPolicyGuardrail` runs `extract_request_tool_names("/v1/messages", request_data)` (`line 151`) and raises `HTTPException(400)`.
   - When `{"type": "bash_20250124", "name": "bash"}` is sent **alongside any standard custom tool** (`{"name": "ping", "input_schema": {"type": "object"}}`), `inputs["tools"]` contains `ping` (`function_tool_names == ["ping"]`). Because `if route and not function_tool_names:` (`line 152`) is `False`, `extract_request_tool_names("/v1/messages", request_data)` is skipped, and `{"type": "bash_20250124", "name": "bash"}` is preserved in `data["tools"]` (`handler.py:722`) without raising `HTTPException(400)`.

## Repository Layout

- `test_tool_policy_reproduction.py` — Standalone `pytest` suite with 3 passing control assertions and 3 failing end-to-end reproduction assertions.
- `evidence/submission-reproduction.log` — Captured test execution output against commit `e63fc85b7b6db8ea1bc0f444f3b997de71898b66`.
- `evidence/tool-policy-guardrail-suite.log` — Captured upstream `tests/unit/proxy/guardrails/guardrail_hooks/test_tool_policy_guardrail.py` baseline output (`11 passed`) on the same checkout.

## How to Run Against a `BerriAI/litellm` Checkout

1. Clone `BerriAI/litellm` and check out the pinned commit:

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
git checkout e63fc85b7b6db8ea1bc0f444f3b997de71898b66
```

2. Copy `test_tool_policy_reproduction.py` into `tests/unit/proxy/guardrails/guardrail_hooks/`:

```bash
cp /path/to/litellm-tool-policy-reproduction/test_tool_policy_reproduction.py \
  tests/unit/proxy/guardrails/guardrail_hooks/test_tool_policy_reproduction.py
```

3. Run `pytest`:

```bash
pytest tests/unit/proxy/guardrails/guardrail_hooks/test_tool_policy_reproduction.py -v --tb=short
```

## Expected vs. Observed Results

| Test Case | Scenario | Expected | Observed | Status |
| :--- | :--- | :--- | :--- | :--- |
| `test_control_openai_format_blocks_trusted_tool_after_untrusted_tool_output` | OpenAI `tool_calls` + `role: "tool"` with `read_external_note` (`untrusted`) -> `send_summary` (`trusted`) | Raises `HTTPException(400)` | Raises `HTTPException(400)` | PASS (control) |
| `test_control_openai_format_allows_trusted_tool_after_trusted_tool_output` | OpenAI `tool_calls` + `role: "tool"` with `ping` (`trusted`) -> `send_summary` (`trusted`) | Allowed | Allowed | PASS (control) |
| `test_control_anthropic_handler_blocks_solo_provider_native_bash_tool` | `AnthropicMessagesHandler.process_input_messages` on `/v1/messages` with solo `{"type": "bash_20250124", "name": "bash"}` (`blocked`) | Raises `HTTPException(400)` | Raises `HTTPException(400)` | PASS (control) |
| `test_repro_anthropic_handler_output_response_blocks_trusted_tool_after_untrusted_tool_result` | `AnthropicMessagesHandler.process_output_response` on `/v1/messages` after `tool_result` from `read_external_note` (`untrusted`) -> `send_summary` (`trusted`) | Raises `HTTPException(400)` | Allowed (no exception) | **FAIL (repro)** |
| `test_repro_responses_handler_output_response_blocks_trusted_tool_after_untrusted_output` | `OpenAIResponsesHandler.process_output_response` on `/v1/responses` after `function_call_output` from `read_external_note` (`untrusted`) -> `send_summary` (`trusted`) | Raises `HTTPException(400)` | Allowed (no exception) | **FAIL (repro)** |
| `test_repro_anthropic_handler_blocks_provider_native_bash_tool_when_mixed_with_custom_tool` | `AnthropicMessagesHandler.process_input_messages` on `/v1/messages` with `ping` (`untrusted`) + `{"type": "bash_20250124", "name": "bash"}` (`blocked`) | Raises `HTTPException(400)` | Allowed (no exception) | **FAIL (repro)** |

## Sanitization Note

Machine-specific local checkout paths in `evidence/*.log` were replaced with `<workspace>/litellm` and `<venv>`. No assertions, stack frames, or test outcomes were modified.

## Scope

This repository contains reproduction tests and execution logs only. No production fix is included.

Prepared with AI assistance. The reported tests were executed locally against commit `e63fc85b7b6db8ea1bc0f444f3b997de71898b66`.
