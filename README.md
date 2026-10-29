# LiteLLM `ToolPolicyGuardrail` Security & Trust-Chain Bypass Reproduction

Executable reproduction and root-cause analysis for three bypasses in `BerriAI/litellm`'s `ToolPolicyGuardrail` (`litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py`).

## Summary of Findings

LiteLLM's `ToolPolicyGuardrail` enforces two security invariants:
- **Stage 1 (`input_policy="blocked"`):** Reject requests or responses invoking blocked tools.
- **Stage 2 (`input_policy="trusted"` / `output_policy="untrusted"`):** Prevent cross-tool prompt injection / taint propagation by blocking `input_policy="trusted"` tool calls when the conversation contains output from an `output_policy="untrusted"` tool.

On `main`, three concrete bypasses allow blocked or tainted tool executions to pass through unblocked (`HTTP 200`):

### 1. Complete Trust-Chain (`Stage 2`) Bypass on Anthropic `/v1/messages` & OpenAI `/v1/responses` (`lines 92–112`, `204–221`)
`_resolve_tool_names_from_messages()` only inspects `msg.get("role") == "assistant"` with `msg.get("tool_calls")`, and the Stage 2 taint loop only checks `msg.get("role") == "tool"` with `msg.get("tool_call_id")`.
- On **Anthropic `/v1/messages`**, assistant tool invocations live inside `msg["content"]` as `{"type": "tool_use", "id": "toolu_...", "name": "..."}` and tool results arrive in `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_...", ...}]}`.
- On **OpenAI `/v1/responses`**, tool calls and outputs are passed in `request_data["input"]` as `function_call` and `function_call_output` items.
- Because `_resolve_tool_names_from_messages()` returns `{}` and `msg.get("role") != "tool"` skips `role="user"` `tool_result` blocks, `untrusted_sources` is always empty (`[]`), allowing untrusted tool outputs to trigger `input_policy="trusted"` tools without raising `HTTPException(400)`.

### 2. Request-Side Mixed Tool Format Shadowing (`lines 143–154`)
In `apply_guardrail` (`input_type == "request"`):
```python
tool_names = (
    extract_request_tool_names(route, request_data)
    if route and not function_tool_names
    else [*function_tool_names, *replayed_tool_names]
)
```
Because `extract_request_tool_names(route, request_data)` is gated behind `if route and not function_tool_names`, any request on `/v1/responses` or `/v1/messages` that includes even a single benign OpenAI-format function tool in `inputs["tools"]` (making `function_tool_names` non-empty) skips `extract_request_tool_names` entirely. Blocked MCP server tools (`{"type": "mcp", "server_label": "..."}`) in `request_data["tools"]` are never checked against `registry.get_effective_policies()`.

### 3. Same-Turn Parallel Tool-Call Trust-Chain Bypass (`lines 156–221`)
When an assistant response emits an `output_policy="untrusted"` tool call and an `input_policy="trusted"` tool call in the same turn (`inputs["tool_calls"]`), Stage 2 only inspects prior `role == "tool"` messages in `request_data["messages"]` and allows both calls to execute concurrently.

## Run Reproduction

```bash
python3 reproduce_bug.py
```

## Execution Evidence

```text
==============================================================================
LITELLM ToolPolicyGuardrail BYPASS REPRODUCTION SUITE
==============================================================================

[Case 1] Anthropic /v1/messages Trust-Chain Taint (tool_use -> tool_result -> trusted tool)
  Upstream main : ALLOWED (BYPASS REPRODUCED)
  Patched/ADOM  : BLOCKED (['fetch_untrusted_webpage'] -> ['execute_wire_transfer'])

[Case 2] Request-Side Mixed Tool Shadowing (benign function tool hides blocked MCP tool)
  Upstream main : ALLOWED (BYPASS REPRODUCED)
  Patched/ADOM  : BLOCKED (blocked_tools=['drop_production_db'])

[Case 3] Same-Turn Parallel Tool-Call Taint Bypass ([untrusted_tool, trusted_tool])
  Upstream main : ALLOWED (BYPASS REPRODUCED)
  Patched/ADOM  : BLOCKED (['fetch_untrusted_webpage'] -> ['execute_wire_transfer'])
==============================================================================
```
