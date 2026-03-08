# Dexter Wakeword WASM Architecture (v1)

## Objective

Implement reliable browser-side wakeword detection for three custom models:

- `dexter_start.onnx`
- `dexter_stop.onnx`
- `dexter_abort.onnx`

using `openWakeWord` model pipeline semantics, ONNX Runtime Web, and a modular frontend architecture that keeps wakeword complexity isolated from app UI/business logic.

## Scope

In scope:

- Real-time wakeword inference in browser.
- Event output for Dexter app control (`start_capture`, `stop_capture`, `abort_capture`).
- Per-word thresholds, cooldowns, and basic false-positive controls.
- Deterministic module boundaries for maintainability.

Out of scope (for this phase):

- Backend wakeword fallback.
- Multi-mic beamforming/noise suppression pipelines.
- Training pipeline for new wakeword models.

## First-Principles Constraints

1. `openWakeWord` is a multi-stage streaming pipeline, not one classifier call.
2. Audio must be 16kHz mono float32 and processed in 1280-sample chunks (80ms).
3. The browser must replicate Python preprocessing assumptions or scores collapse.
4. Temporal state is mandatory:
   - rolling mel buffer
   - rolling embedding buffer
5. Runtime outputs are noisy and bursty, so detection logic must include gating/cooldowns.

## Canonical Pipeline

Per 80ms chunk:

1. `melspectrogram.onnx` on `[1, 1280]` float32 audio.
2. Apply transform to mel output: `(x / 10.0) + 2.0`.
3. Push 5 mel frames into mel ring buffer.
4. While mel buffer has >= 76 frames:
   - take 76x32 window.
   - run `embedding_model.onnx` on `[1, 76, 32, 1]`.
   - push 96-d embedding to embedding ring buffer (length 16).
   - run each wakeword classifier on `[1, 16, 96]`.
   - slide mel buffer by 8 frames.

Models for Dexter:

- shared preprocessing models:
  - `melspectrogram.onnx`
  - `embedding_model.onnx`
- wakeword heads:
  - `dexter_start.onnx`
  - `dexter_stop.onnx`
  - `dexter_abort.onnx`

## Architecture Pattern

Do not expose the pipeline to app code directly. Wrap it as an event engine.

### Module boundary

1. `frontend/wakeword/openwakeword_engine.js`
- Stateful inference engine.
- Owns ONNX sessions, mel/embedding buffers, thresholds, cooldowns.
- Receives normalized 1280-sample chunks.
- Emits `score` and `hit` events.

2. `frontend/wakeword/audio-worklet-processor.js`
- Captures audio frames in render thread.
- Produces stable 1280-sample float32 chunks.

3. `frontend/wakeword/audio_capture_controller.js`
- Manages `getUserMedia`, `AudioContext`, `AudioWorkletNode`.
- Handles resampling path to 16kHz if device rate differs.
- Forwards chunks to inference worker.

4. `frontend/wakeword/inference_worker.js`
- Runs ONNX inference off main thread.
- Hosts `openwakeword_engine` instance.
- Sends back `score`, `hit`, and `error` events.

5. `frontend/wakeword/wakeword_openwakeword_adapter.js`
- Public app-facing facade.
- API: `init()`, `start()`, `stop()`, `dispose()`, `on(event, cb)`.
- Converts wakeword hits into app actions.

6. `frontend/app.js`
- Subscribes to adapter events.
- No direct inference details.
- Only maps hit labels to Dexter app control flow.

## Event Contract

Adapter -> app:

- `ready`
  - `{ sampleRate: 16000 }`
- `score`
  - `{ label, score, ts }`
- `hit`
  - `{ label, score, ts }`
- `error`
  - `{ code, message, detail? }`

Recommended label mapping:

- `dexter_start` -> emit `start_capture`
- `dexter_stop` -> emit `stop_capture`
- `dexter_abort` -> emit `abort_capture`

## Detection Policy

Use per-label policy, not one global threshold.

Example initial config:

- `dexter_start`: threshold `0.55`, cooldown `1000ms`
- `dexter_stop`: threshold `0.60`, cooldown `1000ms`
- `dexter_abort`: threshold `0.60`, cooldown `1000ms`

Add optional patience gate:

- require score above threshold for `N` consecutive inference windows (`N=2` default)

Hit criteria:

1. score > threshold
2. past cooldown for same label
3. optional speech gate (VAD) passes if enabled

## VAD Strategy

VAD is optional in v1, but architecture should support it.

If enabled:

- run `silero_vad.onnx` in same worker pipeline.
- maintain hangover frames (for example 10-12 frames).
- treat VAD as confirmation gate, not as trigger to run classifier.

Critical rule:

- Wakeword inference runs continuously to preserve temporal buffers.

## Runtime and Performance Policy

Execution providers:

- default: `wasm`
- optional: `webgpu` for classifier sessions only after stability

Session split strategy:

- Keep `melspectrogram` + `embedding` on `wasm` for compatibility.
- Consider `webgpu` for wakeword classifier heads if profiling proves stable.

Memory/latency safeguards:

- Reuse typed arrays where possible.
- Copy ONNX output views before buffering (avoid alias/reuse bugs).
- Cap chart/debug history in UI.

Expected timing target:

- end-to-end per 80ms chunk under 80ms on target laptop browser.

## Asset Layout

Use static frontend model assets explicitly.

```text
frontend/
  wakeword/
    wakeword_openwakeword_adapter.js
    inference_worker.js
    openwakeword_engine.js
    audio_capture_controller.js
    audio-worklet-processor.js
  models/
    openwakeword/
      melspectrogram.onnx
      embedding_model.onnx
      dexter_start.onnx
      dexter_stop.onnx
      dexter_abort.onnx
      silero_vad.onnx   # optional
```

Do not pull models dynamically at runtime for core app boot. Keep startup deterministic.

## Integration With Current Dexter State Machine

Current app state machine:

- `idle -> listening -> transcribing -> thinking -> speaking -> idle`

Wakeword integration rule:

- Wakeword engine runs while frontend is active.
- App ignores `dexter_start` if already in `listening`.
- App ignores `dexter_stop` if not in `listening`.
- App always accepts `dexter_abort` and resets capture state.

This avoids duplicate transitions and race conditions.

## Failure Handling

Hard failures:

- model load failure
- worklet init failure
- mic permission denied

Behavior:

- emit `error` event
- keep app usable for manual debug controls
- show explicit UI diagnostics

Soft failures:

- intermittent inference exceptions

Behavior:

- skip frame, increment error counter, continue
- circuit-break to `error` after threshold (for example 20 consecutive failures)

## Why this architecture

1. Contains complexity in a standalone wakeword subsystem.
2. Keeps main app code event-driven and simple.
3. Preserves portability when thresholds/models change.
4. Matches known-working `openWakeWord` browser semantics validated in reference code.
5. Supports incremental rollout:
   - first local worker + scores
   - then hits
   - then app state coupling

## Implementation Phases

1. Phase A: Engine parity
- Build worker + engine with shared models + one wakeword model.
- Validate score non-zero and meaningful.

2. Phase B: Three-model detection
- Add `start/stop/abort` heads.
- Add thresholds + cooldowns + patience.

3. Phase C: App integration
- Hook `hit` events to existing Socket.IO control events.
- Add guard logic with app state.

4. Phase D: Tuning
- Add score telemetry panel.
- Calibrate per-word thresholds from recorded sessions.

## Methodical Implementation Plan (WAV-First, Test-First)

Before live mic integration, implement and validate against fixed WAV fixtures:

- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_start.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_start2.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_stop.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_stop2.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_abort.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_abort2.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/unrelated.wav`
- `/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/unrelated2.wav`

Test-first sequence:

1. Implement offline WAV runner for the exact browser pipeline (resample -> frame -> mel -> embedding -> classifier).
2. Add deterministic test that asserts non-zero, rising score curves on positive files.
3. Add deterministic test that asserts lower peaks on unrelated files.
4. Add per-label peak reporting (`max_score`) and frame index of peak.
5. Gate progression to mic/worklet integration on WAV test pass.

Directional acceptance criteria for first successful integration:

- Positive file should show peak around `>= 0.4` for its intended label (directionally correct target, not final threshold).
- Unrelated files should remain below the positive peak distribution in most runs.
- If not met, iterate on preprocessing parity first (chunking, tensor shapes, mel transform, window slide), not threshold tuning.

Implementation discipline:

- Implement one pipeline stage at a time.
- Add/adjust tests immediately after each stage.
- Do not move to next stage until current stage test output is understood.
- Only after offline corpus behavior looks correct, enable real-time mic path.

## Benchmark Baseline (Source of Truth)

Use this file as the canonical Python baseline for browser parity checks:

- `/Users/davidjvitale/workspace/dexter/ai_docs/benchmark.json`

Implementation rule:

- Any JS/WASM pipeline change must be compared against this baseline before integration into app control flow.

Current baseline summary from `benchmark.json`:

- Positive set winner-by-peak: 6/6 correct (`start`, `stop`, `abort` all matched expected labels).
- Positive peak ranges:
  - `dexter_start`: `0.563` to `0.674`
  - `dexter_stop`: `0.414` to `0.702`
  - `dexter_abort`: `0.726` to `0.898`
- Unrelated peaks remained near zero (`<= 0.0009` across all labels).

JS parity acceptance gates (initial):

1. Winner-by-peak on the same 8 WAV fixtures must match Python baseline for all 6 positive files.
2. Positive peaks should be directionally consistent with baseline ranges (allow drift, but no collapse to near-zero).
3. Unrelated files must stay near-zero and at least an order of magnitude below positive peaks.
4. If parity fails, debug in this order:
   - sample rate and chunk size (`16kHz`, `1280`)
   - mel transform (`x/10 + 2`)
   - mel windowing (`76`, slide `8`)
   - embedding history size (`16`)
   - tensor shapes and output-buffer copying

Benchmark maintenance:

- Keep `benchmark.json` under version control.
- Regenerate only when models, preprocessing, or fixture WAVs intentionally change.
- When regenerating, document why and preserve previous benchmark in git history.

## Non-negotiables

- 16kHz mono float32 input.
- 1280-sample chunking.
- mel transform `(x / 10) + 2`.
- rolling windows: mel 76/step 8, embedding 16.
- copy ONNX output buffers before persisting.

If any of these are violated, detection quality is likely to fail silently.
