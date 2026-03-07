# Dexter – Local Voice Assistant (v1 Tech Spec)

## Goal

Build a **very simple local voice assistant prototype** called **Dexter**.

The system should:

- run **fully locally**
- use **browser mic → backend processing**
- support **wake words**
- run **Whisper + local LLM + TTS**
- display simple UI states in a browser

This is **Hello World architecture only**.  
No advanced features yet.

---

# High-Level Architecture

Phone / Browser
↓
Wake Word Detection
↓
Record Audio
↓
WebSocket Stream
↓
Python Backend
↓
Whisper STT
↓
Local LLM
↓
Text Response
↓
TTS Audio
↓
Browser Playback

---

# Wake Words

Use **Porcupine wake word engine** in the browser.

Three phrases:

Dexter start
Dexter stop
Dexter abort

Behavior:

| Phrase | Action |
|------|------|
| Dexter start | begin recording |
| Dexter stop | stop recording and process command |
| Dexter abort | cancel and reset assistant |

Wake word detection runs **locally in the browser**.

Only **active utterances** are sent to the backend.

---

# Backend Models

## LLM

Choose one of these:

LiquidAI/LFM2-24B-A2B-MLX-4bit  (~12GB RAM)
LiquidAI/LFM2-24B-A2B-MLX-5bit  (~16GB RAM)

Run with:

mlx-lm

---

## Speech to Text

Use **Whisper**.

Options:

Whisper Medium
Whisper Large

Prefer:

mlx-whisper

---

## Text to Speech

Use a simple local engine.

Recommended:

Piper TTS

---

# Backend

Language:

Python

Framework:

Flask
Flask-SocketIO

Responsibilities:

- receive audio from frontend
- run Whisper
- run LLM
- run tools (later)
- generate TTS response
- send UI state updates

---

# Frontend

Very minimal web UI.

Responsibilities:

- microphone capture
- wake word detection
- audio streaming
- state visualization

The backend controls UI state.

Example states:

| State | Color |
|------|------|
| idle | gray |
| listening | green |
| thinking | yellow |
| speaking | blue |
| error | red |

---

# WebSocket Protocol

Use a single WebSocket connection.

## Frontend → Backend

Audio streaming:

audio_chunk
start_recording
stop_recording
abort

Example:

```json
{
  "type": "audio_chunk",
  "pcm": "..."
}


⸻

Backend → Frontend

State updates:

{
  "type": "state",
  "value": "listening"
}

Transcript:

{
  "type": "transcript",
  "text": "open my calendar"
}

Response:

{
  "type": "response",
  "text": "Opening calendar"
}


⸻

Audio Format

Browser audio:

16kHz
mono
16-bit PCM

Send chunks every:

20–50 ms


⸻

Basic Assistant Flow

1. Browser loads page
2. User grants mic permission
3. Wake word engine runs locally
4. User says "Dexter start"
5. Browser records audio
6. Audio streamed to backend
7. User says "Dexter stop"
8. Backend runs Whisper
9. Backend runs LLM
10. Backend generates response
11. Backend sends TTS audio
12. Browser plays response

Abort flow:

User says "Dexter abort"
→ reset assistant state


⸻

Memory Budget (24GB Mac)

Approximate usage:

Component	RAM
LLM (4bit)	~12GB
Whisper Large	~5GB
TTS	~0.5GB

Total ≈ 17–18GB

⸻

Running the System

Start backend:

Open frontend:

http://localhost:5000

Grant microphone permission.

Assistant begins listening.

⸻

Future Improvements

Not part of v1:
	•	tool calling
	•	command memory
	•	multi-turn conversations
	•	silence detection
	•	mobile apps
	•	always-on assistants

