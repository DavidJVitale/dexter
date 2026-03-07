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
