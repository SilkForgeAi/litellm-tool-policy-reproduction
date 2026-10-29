# LiteLLM `ToolPolicyGuardrail` Non-OpenAI Message & Mixed-Tool Format Reproduction

Minimal executable reproduction for two message/tool format handling gaps in `BerriAI/litellm`'s `ToolPolicyGuardrail` (`litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py`).

## Pinned Target

- **Repository:** `https://github.com/BerriAI/litellm`
- **Commit:** `e63fc85b7b6db8ea1bc0f444f3b997de71898b66`
- **Package Version (`litellm`):** `1.93.0`
- **Affected File:** `litellm/proxy/guardrails/guardrail_hooks/tool_policy/tool_policy_guardrail.py` (lines `92–112`, `141–155`, and `204–240`)

## Summary

`ToolPolicyGuardrail.apply_guardrail()` enforces two stages of tool policy from `ToolPolicyRegistry`:
- **Stage 1 (`lines 180–199`):** Raises `HTTPException(status_code=400)` when any tool in `tool_names` has effective `input_policy == "blocked"`.
- **Stage 2 (`lines 204–240`, `input_type == "response"`):** Raises `HTTPException(status_code=400)` when a response calls a tool with `input_policy == "trusted"` and the conversation history contains output from a tool with `output_policy == "untrusted"`.

Two format-handling conditions in `tool_policy_guardrail.py` prevent these checks from triggering outside single-format OpenAI Chat Completions requests:

1. **Stage 2 trust-chain evaluation only inspects OpenAI `role == "tool"` messages (`lines 92–112` and `204–221`):**
   - `_resolve_tool_names_from_messages()` (`lines 92–112`) builds `tc_id_to_name` exclusively from `msg.get("tool_calls")` on `role == "assistant"` messages.
   - `apply_guardrail()` (`lines 211–220`) skips any message where `msg.get("role") != "tool"`.
   - In Anthropic `/v1/messages` conversations (where assistant tool calls are recorded as `{"type": "tool_use", "id": "...", "name": "..."}` content blocks and tool results are sent in `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}`), `tc_id_to_name` is `{}` and `untrusted_sources` is `[]`. As a result, a tool configured with `input_policy == "trusted"` is allowed immediately after output from a tool configured with `output_policy == "untrusted"`.

2. **Stage 1 request extraction skips `extract_request_tool_names(route, request_data)` when `function_tool_names` is non-empty (`lines 142–154`):**
   ```python
   function_tool_names: Final = [
       t["function"]["name"]
       for t in tools
       if isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name")
   ]
   route: Final = _get_request_route_from_data(request_data)
   replayed_tool_names: Final = replayed_tool_search_tool_names(_REQUEST_DATA.validate_python(request_data))
   tool_names = (
       extract_request_tool_names(route, request_data)
       if route and not function_tool_names
       else [*function_tool_names, *replayed_tool_names]
   )
   ```
   When a `/v1/responses` or `/v1/messages` request contains both a nested OpenAI-shaped function tool (`{"type": "function", "function": {"name": "ping"}}`) and a flat/route-specific tool (`{"type": "function", "name": "archive_record"}`), `function_tool_names` is non-empty (`["ping"]`), so `extract_request_tool_names(route, request_data)` is skipped and `archive_record` (`input_policy == "blocked"`) does not raise `HTTPException(400)`.

## Repository Layout

- `test_tool_policy_reproduction.py` — Standalone `pytest` suite with 3 passing control assertions and 2 failing reproduction assertions.
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
| `test_control_request_stage_blocks_route_extracted_tool_when_function_tools_empty` | `/v1/messages` request with Anthropic tool schema containing blocked `archive_record` | Raises `HTTPException(400)` | Raises `HTTPException(400)` | PASS (control) |
| `test_repro_anthropic_messages_tool_result_blocks_trusted_tool_call` | Anthropic `/v1/messages` `tool_use` + `role: "user"` `tool_result` (`untrusted`) -> `send_summary` (`trusted`) | Raises `HTTPException(400)` | Allowed (no exception) | **FAIL (repro)** |
| `test_repro_request_stage_blocks_route_tool_when_mixed_with_openai_function_tool` | `/v1/responses` request mixing nested `{"function": {"name": "ping"}}` and flat `{"type": "function", "name": "archive_record"}` (`blocked`) | Raises `HTTPException(400)` | Allowed (no exception) | **FAIL (repro)** |

## Sanitization Note

Machine-specific local checkout paths in `evidence/*.log` were replaced with `<workspace>/litellm` and `<venv>`. No assertions, stack frames, or test outcomes were modified.

## Scope

This repository contains reproduction tests and execution logs only. No production fix is included.

Prepared with AI assistance. The reported tests were executed locally against commit `e63fc85b7b6db8ea1bc0f444f3b997de71898b66`.
