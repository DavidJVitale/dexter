# Wakeword WASM Architecture Notes

## Goal

Define the frontend architecture for running `openWakeWord` in WebAssembly as a standalone module.

## Status

Draft started on March 7, 2026.

## Next

- Define module boundaries (`adapter`, `worker`, `audio worklet`).
- Define event contract (`ready`, `score`, `hit`, `error`).
- Define threshold/cooldown strategy.
- Define integration points with main app state machine.

## Model Location

dexter/models/wakewords

contains 

dexter start
dexter stop
dexter abort


CONTEXT DUMPS

DUMP 1: CONTEXT DUMP FROM ANOTHER AGENT


# openWakeWord Browser Implementation (WASM / ONNX Runtime Web)

## Purpose

This document provides implementation context for running **openWakeWord wake-word detection fully in the browser** using **WebAssembly via ONNX Runtime Web**.

The coding agent should implement **real-time inference using openWakeWord ONNX models**, with microphone input captured from the browser and processed locally.

The implementation target is **client-side wake word detection** that triggers events (e.g., starting voice command recording).

This document focuses on architecture, known working implementations, technical constraints, and practical gotchas.

---

# Reference Implementation

A working browser implementation already exists:

DeepCore Labs:

- https://deepcorelabs.com/open-wake-word-on-the-web/
- https://deepcorelabs.com/projects/openwakeword/

This implementation demonstrates:

- openWakeWord running entirely in the browser
- ONNX Runtime Web executing the models
- WASM and WebGPU execution backends
- microphone audio streamed into the model pipeline

The project confirms the approach is technically viable.

Important observations from that implementation are included below.

---

# openWakeWord Model Architecture

openWakeWord is not a single model. It is a **pipeline composed of multiple ONNX models**.

Typical pipeline structure:

Audio
→ Mel Spectrogram Model
→ Embedding Model
→ Wake Word Classifier

The separation improves streaming performance and allows efficient incremental inference.

Typical ONNX files provided with openWakeWord:

melspectrogram.onnx
embedding_model.onnx
wakeword_classifier.onnx

Each model must be loaded separately and executed sequentially.

The pipeline mirrors the Python implementation in the openWakeWord repository.

---

# Audio Requirements

Input audio format must match the model expectations.

Required format:

Sample rate: 16000 Hz
Channels: mono
Encoding: float32 PCM

Browser microphone input is usually:

48000 Hz

Therefore the pipeline must include **resampling from 48kHz → 16kHz**.

Resampling should occur before model inference.

---

# Frame-Based Processing

openWakeWord operates on **small audio frames**, not entire recordings.

Typical frame duration:

80 milliseconds

Pipeline per frame:

audio frame
→ mel spectrogram
→ embedding model
→ embedding vector
→ append to embedding buffer

Wake-word classification requires a **sequence of embeddings**, not a single embedding.

---

# Embedding Window

The classifier model expects a **sequence of embeddings** representing roughly ~1–1.5 seconds of audio.

DeepCoreLabs discovered the classifier expects tensors shaped as:

[batch, sequence_length, feature_dimension]

Typical parameters:

sequence_length = 16

Implementation flow:

embedding_1
embedding_2
…
embedding_16

Once 16 embeddings are available:

stack embeddings
reshape tensor
run classifier

Then shift the window and repeat.

This is effectively a **sliding temporal window**.

---

# Inference Pipeline

The complete inference loop should look like this:

while microphone active:

frame = next_audio_frame()

mel = run_melspectrogram_model(frame)

embedding = run_embedding_model(mel)

append embedding to rolling buffer

if buffer contains >= 16 embeddings:

    classifier_input = stack_embeddings(buffer)

    prediction = run_classifier_model(classifier_input)

    if prediction > threshold:
        trigger_wakeword_event

The embedding buffer should operate as a **ring buffer** to avoid array reallocation.

---

# Model Loading in Browser

Use **ONNX Runtime Web**.

Recommended package:

onnxruntime-web

Documentation:

https://onnxruntime.ai/docs/get-started/with-javascript/web.html

Load models using `fetch()` or static assets.

Example directory layout:

/models
melspectrogram.onnx
embedding_model.onnx
wakeword_classifier.onnx

Initialize ONNX inference sessions after loading.

---

# Execution Providers

ONNX Runtime Web supports several backends.

Available execution providers:

WASM
WebGPU
WebGL

Recommended default:

WASM

Optional acceleration:

WebGPU

WebGL is less reliable and generally unnecessary.

Example initialization:

executionProviders: [“wasm”]

Or optionally:

executionProviders: [“webgpu”, “wasm”]

---

# Browser Audio Capture

Use the Web Audio API.

Typical architecture:

navigator.mediaDevices.getUserMedia
→ AudioContext
→ AudioWorkletProcessor
→ audio frame buffer

AudioWorklet is preferred over ScriptProcessorNode because ScriptProcessorNode is deprecated and has higher latency.

Frames should be extracted in consistent chunks corresponding to the frame size required by openWakeWord.

---

# Resampling

Browser microphones typically operate at:

44100 Hz or 48000 Hz

The models require:

16000 Hz

Resampling must occur before the mel spectrogram stage.

Options:

- implement a lightweight resampler
- use a WebAudio resampling node
- perform linear interpolation resampling

Accuracy matters because mismatched preprocessing can degrade wake-word detection.

---

# Model Preprocessing Consistency

The mel spectrogram computation must match the original Python implementation exactly.

Key parameters include:

FFT size
hop length
window size
mel bin count
window function

Any mismatch will produce embeddings that the classifier does not recognize.

These parameters should be taken directly from the openWakeWord source implementation.

---

# Classifier Output

The classifier produces a **probability score** representing wake word confidence.

Typical threshold values:

0.4 – 0.7

Lower threshold:

more sensitive
more false positives

Higher threshold:

less sensitive
fewer false positives

The threshold should be configurable.

---

# Performance Expectations

Typical inference time per frame:

WASM: 3–10 ms
WebGPU: 1–3 ms

Full pipeline latency should remain under:

80 ms per frame

This allows real-time wake-word detection.

---

# Worker-Based Architecture (Recommended)

Inference should run in a **Web Worker**.

Suggested architecture:

Main Thread
→ microphone capture
→ send audio frames to worker

Worker
→ run inference pipeline
→ return wake-word detection events

This prevents inference from blocking UI rendering.

---

# State Handling

The wake-word engine should run continuously.

When a wake word is detected:

emit event

The surrounding application should then begin recording voice commands.

Example integration:

wake word detected
→ start recording command
→ send command audio to backend
→ speech-to-text
→ LLM
→ TTS

The wake-word detector should remain active in parallel.

---

# Important Implementation Details

### Rolling Embedding Buffer

Maintain a rolling window of embeddings with fixed length.

Do not recreate arrays each frame.

Use a ring buffer.

---

### Shape Validation

Ensure tensors match expected shape exactly:

[batch, sequence, features]

Incorrect tensor shape was the main issue encountered by the DeepCoreLabs implementation.

---

### Floating Point Precision

Browser inference uses different floating point implementations compared to Python.

However, DeepCoreLabs confirmed that inference results are still valid when tensor shapes and preprocessing are correct.

---

### Audio Normalization

Ensure audio samples are normalized consistently with the Python implementation.

Typical range:

-1.0 to 1.0

---

# Integration With Voice Assistant System

The wake word engine should only trigger events.

Example architecture:

Browser
→ wake word detection
→ start recording command
→ stream command audio to backend

Backend
→ Whisper
→ LLM
→ TTS
→ return response

This prevents continuous microphone streaming to the backend.

---

# Known Limitations

- openWakeWord does not officially maintain a browser runtime
- preprocessing must be implemented carefully
- audio resampling is required
- the model pipeline must replicate the Python implementation

Despite this, successful browser implementations exist and perform well.

---

# Expected End Result

A browser-based wake-word engine that:

- runs fully client-side
- requires no server inference
- requires no API keys
- performs real-time wake-word detection
- triggers events for the main voice assistant pipeline


DUMP 2: BLOG POST FROM SOMEONE WHO DID THIS

Title: Open Wake Word on the Web – Deep Core Labs

URL Source: https://deepcorelabs.com/open-wake-word-on-the-web/

Markdown Content:
[OpenWakeWord - Web Demo](https://deepcorelabs.com/projects/openwakeword)

How I Ported a Python Wake Word System to the Browser When the LLMs Gave Up
---------------------------------------------------------------------------

I started this project with a goal that seemed simple on paper: take [openWakeWord](https://github.com/dscripka/openWakeWord/), a powerful open-source library for wake word detection, and make it run entirely in a web browser. And when I say “in the browser,” I mean it. No tricks. No websockets streaming audio to a Python server. I wanted the models, the audio processing, and the detection logic running completely on the client.

 My initial approach was to “vibe-code” it with the new generation of LLMs. I fed my high-level goal to **Gemini 2.5 Pro, o4-mini-high, and Grok 4**. They gave me a fantastic head start, building out the initial HTML, CSS, and JavaScript structure with impressive speed. But after dozens of messages just refining the vibe, we hit a hard wall. The models would run, but the output score was just a flat line at zero. No errors, no crashes, just… nothing.

 This is where the real story begins. The vibe was off. Vibe coding had failed. I had to pivot from being a creative director to a deep-dive detective. It’s a tale of how I used a novel cross-examination technique with these same LLMs to solve a problem that each one, individually, had given up on.

### TL;DR: The `openWakeWord` JavaScript Architecture That Actually Works

For the engineers who just want the final schematics, here is the stateful, multi-buffer pipeline required to make this work.

*   **Pipeline:**`[Audio Chunk]` ->`Melspectrogram Model` ->`Melspectrogram Buffer` ->`Embedding Model` ->`Wake Word Model` ->`Score`
*   **Stage 1: Audio to Image (Melspectrogram):**
    *   **Audio Source:** 16kHz, 16-bit, Mono PCM audio.
    *   **Chunking:** The pipeline operates on **1280 sample** chunks (80ms). This is non-negotiable.
    *   **Model Input:** The chunk is fed into `melspectrogram.onnx` as a **float32** tensor.
    *   **Mandatory Transformation:** The output from the melspectrogram model **must** be transformed with the formula `output = (value / 10.0) + 2.0`.

*   **Stage 2: Image Analysis (Feature Embedding):**
    *   **Melspectrogram Buffer:** The 5 transformed spectrogram frames from Stage 1 are pushed into a buffer.
    *   **Sliding Window:** This stage only executes when the `mel_buffer` contains at least **76 frames**. A  window is sliced from the _start_ of the buffer.
    *   **Model Input:** This window is fed into `embedding_model.onnx` as a  tensor.
    *   **Window Step:** After processing, the buffer is slid forward by **8 frames** (`splice(0, 8)`).

*   **Stage 3: Prediction:**
    *   **Embedding Buffer:** The 96-value feature vector from Stage 2 is pushed into a second, fixed-size buffer that holds the last **16** embeddings.
    *   **Model Input:** Once full, the 16 embeddings are flattened and fed into the final wake word model as a  tensor. This `[batch, sequence, features]` shape is the critical insight that resolved a key error.

* * *

### The Unvarnished Truth: My Journey into Debugging Hell

After the initial burst of productivity, all three LLMs hit the same wall and gave up. They settled on the same, demoralizing conclusion: the problem was **floating-point precision differences** between Python and the browser’s ONNX Runtime. They suggested the complex math in `openWakeWord` was too sensitive and that a 100% client-side implementation was likely **impossible**.

 Something about that felt fishy. The separate VAD (Voice Activity Detection) model was working perfectly fine. This felt like a logic problem, not a fundamental platform limitation.

 This is where the breakthrough happened. I realized “vibe coding” wasn’t enough. I had to get specific. I decided to change my approach and use the LLMs as specialized, focused tools rather than general-purpose partners:

1.   **The Analyst:** I tasked one LLM with a single, focused job: analyze the `openwakeword` Python source code and describe, in painstaking detail, exactly what it was doing at every step.
2.   **The Coder:** I took the detailed blueprint from the “Analyst” and fed it to a _different_ LLM. Its job was to take that blueprint and write the JavaScript implementation.

This cross-examination process was like a magic trick. It bypassed the ruts the models had gotten into and started revealing the hidden architectural assumptions that had been causing all the problems.

#### The First Wall: The Sound-to-Image Pipeline

The “Analyst” LLM immediately revealed my most basic misunderstanding. I thought I was feeding a sound model, but that’s not how it works. These models don’t “hear” sound; they “see” it.

**Aha! Moment #1: It’s an Image Recognition Problem.** The first model in the chain, `melspectrogram.onnx`, doesn’t process audio waves. Its entire job is to convert a raw 80ms audio chunk into a **melspectrogram**—a 2D array of numbers that is essentially an image representing the intensity of different frequencies in that sound. The subsequent models are doing pattern recognition on these sound-images, not on the audio itself. This also explained the second part of the puzzle: the models were trained on specifically processed images, which is why this transformation was mandatory:

```
// This isn't just a normalization; it's part of the "image processing" pipeline
// that the model was trained on. It fails silently without it.
for (let j = 0; j < new_mel_data.length; j++) {
new_mel_data[j] = (new_mel_data[j] / 10.0) + 2.0;
}
```

#### The Second Wall: The Audio History Tax

With the formula in place, my test WAV file still failed. The “Analyst” LLM’s breakdown of the Python code’s looping was the key. I realized the pipeline’s second stage needs a history of **76 spectrogram frames** to even begin its work. Each 80ms audio chunk only produces **5 frames**, meaning the system has to process **16 chunks** (1.28 seconds) of audio before it can even think about generating the first feature vector. My test file was too short.

```
// This logic checks if the audio is long enough and pads it with silence if not.
const minRequiredSamples = 16 * frameSize; // 16 chunks * 1280 samples/chunk = 20480
if (audioData.length < minRequiredSamples) {
const padding = new Float32Array(minRequiredSamples - audioData.length);
const newAudioData = new Float32Array(minRequiredSamples);
newAudioData.set(audioData, 0);
newAudioData.set(padding, audioData.length);
audioData = newAudioData; // Use the new, padded buffer
}
```

#### The Third Wall: The Treachery of Optimization

The system came to life, but it was unstable, crashing with a bizarre `offset is out of bounds` error. This wasn’t a floating-point issue; it was a memory management problem. I discovered that for performance, the ONNX Runtime for web **reuses its memory buffers**. The variable I was saving wasn’t the data, but a temporary _reference_ to a memory location that was being overwritten.

```
// AHA Moment: ONNX Runtime reuses its output buffers. We MUST create a *copy*
// of the data instead of just pushing a reference to the buffer.
const new_embedding_data_view = embeddingOut[embeddingModel.outputNames[0]].data;
const stable_copy_of_embedding = new Float32Array(new_embedding_data_view);
embedding_buffer.push(stable_copy_of_embedding); // Push the stable copy, not the temporary view.
```

#### The Final Wall: The Purpose of the VAD

The system was finally stable, and I could see the chart spike to 1.0 when I spoke the wake word. But the success sound wouldn’t play reliably. This was due to my most fundamental misconception. I had assumed the VAD’s purpose was to save resources. My thinking was: “VAD is cheap, the wake word model is expensive. So, I should only run the expensive model when the VAD detects speech.”

 This is completely wrong.

**Aha! Moment #4: The VAD is a Confirmation, Not a Trigger.** The wake word pipeline must run _continuously_ to maintain its history buffers. The VAD’s true purpose is to act as a **confirmation signal**. A detection is only valid if two conditions are met simultaneously: the wake word model reports a high score, AND the VAD confirms that human speech is currently happening. It’s a two-factor authentication system for your voice. This led to the final race condition: the VAD is fast, but the wake word pipeline is slow. The solution was a **VAD Hangover**—what I call “Redemption Frames”—to keep the detection window open just a little longer.

```
// These constants define the VAD Hangover logic
const VAD_HANGOVER_FRAMES = 12; // Keep speech active for ~1 second after VAD stops
let vadHangoverCounter = 0;
let isSpeechActive = false;

// Later, the final check uses this managed state:
if (score > 0.5 && isSpeechActive) {
// Detection is valid!
}
```

### The Backend Betrayal: A Final Hurdle

With the core logic finally perfected, I implemented a feature to switch between the WASM, WebGL, and WebGPU backends. WASM and WebGPU worked, but WebGL crashed instantly with the error: `Error: no available backend found. ERR: [wasm] backend not found`.

 The issue was that the melspectrogram.onnx model uses specialized audio operators that the WebGL backend in ONNX Runtime simply does not support. My code was trying to force all models onto the selected backend, which is impossible when one is incompatible. The solution was a hybrid backend approach: force the incompatible pre-processing models (melspectrogram and VAD) to run on the universally-supported WASM backend, while allowing the heavy-duty neural network models to run on the user’s selected GPU backend for a performance boost. I’ve left the WebGL option in the demo as a reference for this interesting limitation.

### The Final Product

This journey was a powerful lesson in the limitations of “vibe coding” for complex technical problems. While LLMs are incredible for scaffolding, they can’t replace rigorous, first-principles debugging. By pivoting my strategy—using one LLM to deconstruct the source of truth and another to implement that truth—I was able to solve a problem that a single LLM, or even a committee of them, declared impossible. The result is a working, robust web demo that proves this complex audio pipeline can indeed be tamed, running **100% on the client, in the browser**, no Python backend required.

[OpenWakeWord - Web Demo](https://deepcorelabs.com/projects/openwakeword)

DUMP 3:

dexter/ai_docs/deepcorelabs_openwake_reference_files contains the entire code of that website from above