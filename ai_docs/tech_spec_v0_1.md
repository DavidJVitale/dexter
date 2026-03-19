# Dexter v0.1 Technical Specification

## Status

Draft v0.1 (March 7, 2026)

## Product Intent

Dexter is a local-first voice assistant prototype that runs on a single MacBook Air M3 (24GB RAM) without cloud dependencies.

v0.1 goals:

- Prefer browser wake words (`Dexter start`, `Dexter stop`, `Dexter abort`) through `openWakeWord` via WASM.
- Stream voice audio from browser to backend for active utterances only.
- Run local STT + LLM + TTS.
- Return both response text and response audio to frontend.
- Support repeated single-turn commands while browser session remains open.

Out of scope for v0.1:

- Conversation memory across turns.
- Tool calling and external integrations.
- Mobile-native clients.
- Cloud deployment.
- Push-to-talk fallback.

## Key Constraints and Decisions

- Runtime: local-only on macOS laptop.
- Wake word primary path: `openWakeWord` in browser via WASM.
- Backend must stay wake-engine-agnostic.
- LLM: local MLX 24B 4-bit model (~12-13GB RAM observed).
- LLM models available:
  - `LiquidAI/LFM2-24B-A2B-MLX-4bit` (preferred default for v0.1)
  - `LiquidAI/LFM2-24B-A2B-MLX-5bit` (optional quality-first variant)
- STT: `mlx-whisper` medium to start.
- TTS: local OSS engine (Piper preferred).
- Priority: lower latency over maximum transcription/response quality.

## System Architecture

1. Frontend captures mic audio continuously for wake word detection.
2. `openWakeWord` detects start/stop/abort phrases in browser (WASM path).
3. Frontend sends generic control events and binary audio chunks over Socket.IO.
4. Backend session manager assembles utterance audio.
5. On `stop_capture`, backend runs STT -> LLM -> TTS pipeline.
6. Backend emits:
   - state updates
   - transcript text
   - assistant text response
   - encoded assistant audio payload
7. Frontend displays transcript/response text and plays response audio.

## Backend Modularity Rule (Wake-Word Agnostic)

Backend will not encode wake-engine-specific assumptions. It will only accept generic session control events:

- `start_capture`
- `stop_capture`
- `abort_capture`

This allows replacing or mixing wake-word engines later without backend logic changes.

## Proposed Repo Layout (within `dexter/`)

```text
 dexter/
   api/
     __init__.py
     main.py
     routes/
       health.py
       socket_events.py
     services/
       assistant_orchestrator.py
       stt_service.py
       llm_service.py
       tts_service.py
       audio_codec.py
     models/
       session_state.py
     config.py
   frontend/
     index.html
     app.js
     style.css
     wakeword_openwakeword_adapter.js
     audio_capture.js
     socket_client.js
   scripts/
     run_local_backend.sh
     run_local_frontend.sh
   logs/
   ai_docs/
     initial_architecture.md
     tech_spec_v0_1.md
   pyproject.toml
```

## Runtime Model Lifecycle

To reduce startup and per-turn latency:

- Lazy-load models on first request with optional preload endpoint.
- Keep STT/LLM/TTS loaded in process for session duration.
- Single in-flight pipeline per browser session for v0.1.
- Reject/queue overlapping `stop_capture` events while processing.
- Follow existing local MLX pattern from `local_inference_helpers/mlx`:
  - load once (`mlx_lm.load`) and reuse
  - generate per request (`mlx_lm.generate`) with low-latency defaults
  - expose lightweight timing/memory metrics for tuning

## Tool Use Architecture (Planned)

Dexter will support a configurable set of backend tools that the assistant may call during a single user request. The tool layer must remain model agnostic. Although the current local model is `LiquidAI/LFM2-24B-A2B-MLX-4bit` and its tool-use docs inform prompting strategy, Dexter must not depend on Liquid-specific Python bindings, tokenizer helpers, or special tool-call tokens as the core runtime contract.

### Design Goals

- Support `n` configurable tools without changing the core assistant pipeline.
- Keep the backend tool layer independent from any specific model vendor or prompt template API.
- Use the same abstraction for local tools, shell/CLI-backed tools, and remote passthrough tools.
- Preserve the existing voice UX:
  - user speaks once
  - assistant may call tools
  - assistant returns a short final verbal response

### Standard Tool Abstractions

Backend tools will follow a common registry + executor pattern.

Conceptual Python-side abstractions:

```python
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    safety_mode: Literal["read", "write", "network", "passthrough"]
    enabled: bool


class ToolResult:
    ok: bool
    content: dict | list | str
    error: str | None


class ToolExecutor:
    def list_tools(self, context) -> list[ToolSpec]: ...
    def execute(self, tool_name: str, arguments: dict, context) -> ToolResult: ...
```

Notes:

- `ToolSpec` is the canonical metadata exposed to the model.
- `input_schema` should use a JSON-schema-like shape so tool arguments are model agnostic.
- `ToolResult` is the normalized result shape returned by all tools.
- `ToolExecutor` is the common execution interface regardless of whether the underlying tool is:
  - pure Python logic
  - a shell or CLI command
  - a Google Workspace CLI wrapper
  - an external LLM provider adapter

### Canonical Invocation Contract

Dexter's internal tool invocation protocol will be JSON-first.

The model will be prompted with:

- a system instruction
- a compact JSON list of currently active tool definitions
- the user request
- an instruction to output either:
  - a final answer, or
  - a tool call JSON object

Canonical tool call envelope:

```json
{
  "type": "tool_call",
  "tool_name": "string",
  "arguments": {}
}
```

Canonical final response envelope:

```json
{
  "type": "final_response",
  "content": "string"
}
```

Rules:

- Dexter will not require model-native special tokens for core tool use.
- Liquid-specific tool syntax may be supported later through an adapter layer, but the internal runtime contract remains JSON.
- Prompts may explicitly instruct the model to "output JSON only" when tool selection is expected.

### Tool Execution Loop

Tool use follows a bounded loop:

1. Build the prompt with the current conversation state and only the relevant enabled tools.
2. Generate model output.
3. Parse output as either:
   - `final_response`
   - `tool_call`
4. If the output is a tool call:
   - validate `tool_name`
   - validate `arguments` against the tool schema
5. Execute the tool externally through the tool executor.
6. Append the normalized tool result to the assistant context.
7. Re-run the model so it can interpret the tool result.
8. Stop when the model emits `final_response` or the tool-step limit is reached.

Defaults:

- one tool call per generation step
- maximum `3` tool steps per user request
- if parsing fails, treat output as a normal assistant response
- if tool execution fails, append the failure result once and allow the model to recover or answer
- the final text response continues through the normal TTS path

### Per-Request Working Memory

Dexter should keep a small working memory for the current user request only. This is not intended to be long-term conversational memory. Its purpose is to help the model keep track of what has already been attempted during a single tool-use flow.

The working memory should contain:

- the original user request
- the current tool-step ledger for this request
- short normalized summaries of tool results

This working memory should allow the model to:

- understand what tools have already been called
- avoid redundant tool calls
- reason about whether enough information has been collected
- produce a final answer that reflects what actually happened

For small local models, the working memory should remain compact. Dexter should prefer a structured tool ledger over replaying long raw transcripts, raw CLI output, or full unfiltered tool payloads.

Conceptual structure:

```json
{
  "user_request": "What is on my calendar tomorrow afternoon?",
  "steps": [
    {
      "tool": "google_calendar_readonly",
      "arguments": {
        "range": "tomorrow afternoon"
      },
      "result_summary": "Found 3 events between 1 PM and 5 PM."
    }
  ]
}
```

Policy:

- working memory exists only for the current request
- it should be cleared after the assistant produces a final response
- tool results should be summarized into compact structured form before being returned to the model
- raw tool logs should not be included unless needed for debugging or error recovery
- this phase does not include persistent user memory or cross-request memory

### Tool Visibility and Prompt Budget

Tool definitions consume prompt tokens, so Dexter should not always expose every configured tool.

Policy:

- the backend may maintain many registered tools
- only enabled and relevant tools should be exposed for a given request
- relevance may initially be selected through lightweight heuristics or routing rules
- the architecture must still function if all enabled tools are exposed, but that is not the preferred long-term path

### Safety Model

Each tool must declare a safety posture.

Initial safety rules:

- `calculator` is local and read-only
- `google_calendar_readonly` must never perform writes
- Google Workspace CLI integration for this mode must not be configured for any create/update/delete operations
- `external_llm_passthrough` must only run when the user explicitly requests passthrough
- external passthrough must not trigger implicitly just because the model prefers it
- tool results should be returned to the model in normalized structured form, not raw logs unless needed for error handling

### Initial Tool Families

Dexter will start with these initial tool families.

#### 1. `calculator`

Purpose:

- deterministic arithmetic and simple expression evaluation

Input shape:

- expression string

Output shape:

- normalized expression
- computed result

Properties:

- no network
- no side effects
- appropriate for math, unit conversion, and quick deterministic evaluation

#### 2. `google_calendar_readonly`

Purpose:

- read calendar data through a backend adapter

Allowed operations in this phase:

- list upcoming events
- query events in a date or time range
- inspect a day's agenda
- answer availability-style read questions based on retrieved events

Properties:

- implementation may use the Google Workspace CLI (`gws`) internally
- the Dexter tool abstraction must remain generic and not expose raw CLI details to the model
- strictly read-only for this phase
- no event creation, edits, deletes, RSVPs, or any write actions

#### 3. `external_llm_passthrough`

Purpose:

- send a user-requested prompt to an explicitly selected external LLM provider

This is a single generic passthrough tool, not one tool per provider.

Input shape should include at least:

- `provider`
- `prompt`
- optional `model`

Behavior rules:

- only valid when the user explicitly requests passthrough
- example invocation style:
  - "Dexter, ChatGPT API call passthrough: ..."
  - "Dexter, Claude passthrough: ..."
  - "Dexter, DeepSeek passthrough: ..."
- provider aliases may map to configured backends, but the core tool abstraction remains generic
- output should include:
  - provider name
  - optional model name
  - returned text

### Prompting Guidance for the Current Local Model

The current local model is `LiquidAI/LFM2-24B-A2B-MLX-4bit`.

For tool-enabled prompting, Dexter should prefer:

- a short system instruction
- concise tool definitions
- explicit instruction to output valid JSON only when making a tool decision
- one-tool-at-a-time behavior
- short final answers suitable for TTS

Recommended system guidance for tool-enabled turns should communicate:

- answer directly when no tool is needed
- if a tool is needed, emit only the JSON tool-call envelope
- never invent tools
- never use `external_llm_passthrough` unless the user explicitly asked for passthrough
- after tool results are available, produce a short spoken-friendly final answer

### Non-Goals for This Phase

This section does not yet define:

- full tool implementation details
- OAuth setup steps
- dynamic permission grants
- concurrent multi-tool execution
- write-capable Google integrations
- long-term conversational memory across tool calls

## Python Environment and Model Cache Policy

- Dexter will use its own dedicated virtual environment at `dexter/.venv`.
- Dexter should not reuse the standalone `mlx_env` as its runtime environment.
- Model weights remain shared through Hugging Face cache, so creating a new venv does not duplicate model files.
- Default cache location observed:
  - `HF_HOME=/Users/davidjvitale/.cache/huggingface`
  - `HF_HUB_CACHE=/Users/davidjvitale/.cache/huggingface/hub`
- If needed, set these env vars in Dexter run scripts to force shared cache usage across environments.

## Assistant State Machine

Canonical backend-driven states:

- `idle`
- `listening`
- `transcribing`
- `thinking`
- `speaking`
- `aborted`
- `error`

Transitions:

1. `idle -> listening` on `start_capture`
2. `listening -> transcribing` on `stop_capture`
3. `transcribing -> thinking` when transcript is ready
4. `thinking -> speaking` when text response is ready
5. `speaking -> idle` when audio payload sent
6. `* -> aborted` on `abort_capture`, then `aborted -> idle`
7. `* -> error -> idle` on unrecoverable failure

## Socket.IO Event Contract (v0.1)

Use one Socket.IO connection per browser tab.

Frontend -> Backend:

- `start_capture`
  - payload: `{ "session_id": "uuid", "request_id": "uuid" }`
- `audio_chunk`
  - binary payload + metadata: `{ "request_id": "uuid", "seq": 12, "sample_rate": 16000, "channels": 1, "format": "pcm16" }`
- `stop_capture`
  - payload: `{ "request_id": "uuid" }`
- `abort_capture`
  - payload: `{ "request_id": "uuid", "reason": "wake_word_abort" }`

Backend -> Frontend:

- `state`
  - `{ "request_id": "uuid", "value": "listening" }`
- `transcript`
  - `{ "request_id": "uuid", "text": "hello world" }`
- `response_text`
  - `{ "request_id": "uuid", "text": "Hello. How can I help?" }`
- `response_audio`
  - `{ "request_id": "uuid", "mime": "audio/wav", "b64": "..." }`
- `error`
  - `{ "request_id": "uuid", "code": "STT_FAILED", "message": "..." }`

Notes:

- Return both `response_text` and `response_audio` for every successful turn.
- Audio chunks should be sent every 20-40ms for lower latency.
- For v0.1, request/response is single-turn and stateless beyond in-memory session buffers.

## Audio I/O Requirements

Input from browser:

- PCM 16-bit
- 16kHz
- mono
- chunk interval: 20-40ms

Output to browser:

- WAV or MP3 payload encoded as base64 in `response_audio` event (v0.1 simplicity).
- Frontend decodes and plays via Web Audio or HTMLAudioElement.

## Error Handling and Recovery

- `abort_capture` clears buffered audio and cancels in-flight pipeline if possible.
- If pipeline cannot cancel immediately, backend marks request aborted and suppresses stale outputs.
- Any stage failure emits `error` then returns state to `idle`.
- Include `request_id` in all events for correct UI correlation.

## Observability

- Structured logs to file + stdout.
- Per-turn latency markers:
  - capture duration
  - STT ms
  - LLM ms
  - TTS ms
  - end-to-end ms
- Log model load time and memory footprint estimates at startup.

## Performance Targets (Initial)

Local laptop targets for short utterances (<8s):

- STT completion: <= 1.5s median
- LLM first token: <= 1.2s median
- TTS generation: <= 0.8s median
- end-to-end stop-to-audio: <= 4.0s median

These are directional targets for iteration, not strict SLOs.

## Security and Privacy (v0.1)

- Localhost-only binding by default.
- No cloud API calls for inference path.
- No transcript persistence unless explicitly enabled.
- Do not log raw audio payloads.

## Incremental Implementation Plan

1. Milestone 1: App skeleton + Socket.IO + frontend state UI + stubbed pipeline.
2. Milestone 2: Real STT (`mlx-whisper medium`) + transcript event.
3. Milestone 3: Real LLM response generation + response text event.
4. Milestone 4: Real TTS + response audio event playback.
5. Milestone 5: `openWakeWord` WASM browser integration + abort correctness + latency tuning.

## Acceptance Criteria for v0.1

1. User can run app locally and open frontend at localhost.
2. Saying `Dexter start ... Dexter stop` yields transcript, text response, and audio response.
3. Saying `Dexter abort` reliably resets to `idle` without stale playback.
4. User can run multiple independent single-turn commands in one browser session.
5. No memory carry-over between turns in backend logic.

## Next Step after this Spec

Implement Milestone 1 only:

- Scaffold Flask app and Socket.IO events.
- Build minimal frontend state indicator and transcript/response panes.
- Stub STT/LLM/TTS responses to validate event flow before model wiring.
