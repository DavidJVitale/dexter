import { AudioCapture } from "./audio_capture.js";
import { createSocketClient } from "./socket_client.js";
import { OpenWakeWordAdapter } from "./wakeword_openwakeword_adapter.js";

const statePill = document.getElementById("state-pill");
const requestIdEl = document.getElementById("request-id");
const transcriptEl = document.getElementById("transcript");
const responseTextEl = document.getElementById("response-text");
const responseAudioEl = document.getElementById("response-audio");
const eventLogEl = document.getElementById("event-log");
const debugLogsEnabledCheckbox = document.getElementById("debug-logs-enabled");

const enableMicButton = document.getElementById("enable-mic");
const startButton = document.getElementById("start-capture");
const stopButton = document.getElementById("stop-capture");
const abortButton = document.getElementById("abort-capture");

let currentRequestId = null;
let currentState = "idle";
let latestResponseText = "";
let captureStartedAtMs = 0;
let speechDetected = false;
let lastSpeechAtMs = 0;
const NON_DEBUG_LOG_TYPES = new Set([
  "wakeword_init",
  "wakeword_ready",
  "mic",
  "wakeword_hit",
  "wakeword_error",
]);
const AUTO_STOP_MIN_CAPTURE_MS = 500;
const AUTO_STOP_MAX_CAPTURE_MS = 8000;
const AUTO_STOP_SILENCE_MS = 1200;
const SPEECH_RMS_THRESHOLD = 0.02;

function makeRequestId() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `req_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}

function logEvent(type, payload) {
  const debugEnabled = Boolean(debugLogsEnabledCheckbox?.checked);
  if (!debugEnabled && !NON_DEBUG_LOG_TYPES.has(type)) {
    return;
  }
  const line = `${new Date().toISOString()} ${type} ${JSON.stringify(payload)}`;
  eventLogEl.textContent = `${line}\n${eventLogEl.textContent}`;
}

function setState(state) {
  currentState = state;
  statePill.textContent = state;
  statePill.className = `state ${state}`;
}

function beginCapture(trigger = "manual") {
  if (currentState === "listening") {
    return;
  }
  currentRequestId = makeRequestId();
  requestIdEl.textContent = currentRequestId;
  transcriptEl.textContent = "(none)";
  responseTextEl.textContent = "(none)";
  captureStartedAtMs = Date.now();
  speechDetected = false;
  lastSpeechAtMs = captureStartedAtMs;

  audioCapture.startCapture();
  socket.emit("start_capture", { session_id: socket.id, request_id: currentRequestId });
  logEvent("start_capture", { request_id: currentRequestId, trigger });
}

function stopCapture(trigger = "manual") {
  if (!currentRequestId || currentState !== "listening") {
    return;
  }
  audioCapture.stopCapture();
  socket.emit("stop_capture", { request_id: currentRequestId });
  logEvent("stop_capture", { request_id: currentRequestId, trigger });
}

function abortCapture(trigger = "manual") {
  if (!currentRequestId) {
    return;
  }
  audioCapture.stopCapture();
  socket.emit("abort_capture", { request_id: currentRequestId, reason: "wakeword_abort" });
  logEvent("abort_capture", { request_id: currentRequestId, trigger });
}

function maybeAutoStopCapture(chunkPayload) {
  if (!currentRequestId || currentState !== "listening") {
    return;
  }
  const nowMs = Date.now();
  const captureDurationMs = nowMs - captureStartedAtMs;
  const rms = Number(chunkPayload?.rms || 0);

  if (rms >= SPEECH_RMS_THRESHOLD) {
    speechDetected = true;
    lastSpeechAtMs = nowMs;
  }

  if (captureDurationMs >= AUTO_STOP_MAX_CAPTURE_MS) {
    stopCapture("max_duration");
    return;
  }

  if (!speechDetected || captureDurationMs < AUTO_STOP_MIN_CAPTURE_MS) {
    return;
  }

  if (nowMs - lastSpeechAtMs >= AUTO_STOP_SILENCE_MS) {
    stopCapture("silence");
  }
}

function speakFallbackText(text) {
  const utteranceText = String(text || "").trim();
  if (!utteranceText || !window.speechSynthesis) {
    return;
  }
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(utteranceText));
    logEvent("response_audio_fallback_tts", { text_len: utteranceText.length });
  } catch (error) {
    logEvent("response_audio_fallback_tts_error", {
      message: error?.message || String(error),
    });
  }
}

const socket = createSocketClient();
const wakeword = new OpenWakeWordAdapter();
const audioCapture = new AudioCapture({
  onChunk: (chunkPayload) => {
    if (!currentRequestId) {
      return;
    }
    socket.emit("audio_chunk", { ...chunkPayload, request_id: currentRequestId });
    maybeAutoStopCapture(chunkPayload);
  },
});

socket.on("connect", () => {
  logEvent("connect", { id: socket.id });
});

socket.on("state", (payload) => {
  if (payload.request_id) {
    currentRequestId = payload.request_id;
    requestIdEl.textContent = payload.request_id;
  }
  setState(payload.value);
  logEvent("state", payload);
});

socket.on("transcript", (payload) => {
  transcriptEl.textContent = payload.text || "(none)";
  logEvent("transcript", payload);
});

socket.on("response_text", (payload) => {
  latestResponseText = payload.text || "";
  responseTextEl.textContent = latestResponseText || "(none)";
  logEvent("response_text", payload);
});

socket.on("response_audio", (payload) => {
  responseAudioEl.src = `data:${payload.mime};base64,${payload.b64}`;
  responseAudioEl.play().catch((error) => {
    logEvent("response_audio_play_error", {
      request_id: payload.request_id,
      mime: payload.mime,
      message: error?.message || String(error),
    });
    speakFallbackText(latestResponseText);
  });
  logEvent("response_audio", { request_id: payload.request_id, mime: payload.mime });
});

socket.on("error", (payload) => {
  logEvent("error", payload);
});

enableMicButton.addEventListener("click", async () => {
  await audioCapture.initialize();
  await wakeword.start();
  logEvent("mic", { status: "enabled" });
});

wakeword.on("ready", (payload) => {
  logEvent("wakeword_ready", payload);
});

wakeword.on("hit", (payload) => {
  logEvent("wakeword_hit", payload);
  if (payload.label === "dexter") {
    beginCapture("wakeword");
  }
});

wakeword.on("trace", (payload) => {
  logEvent("wakeword_trace", payload);
});

wakeword.on("error", (payload) => {
  logEvent("wakeword_error", payload);
});

startButton.addEventListener("click", () => {
  beginCapture("manual");
});

stopButton.addEventListener("click", () => {
  stopCapture("manual");
});

abortButton.addEventListener("click", () => {
  abortCapture("manual");
});

async function initializeOnPageLoad() {
  try {
    await wakeword.initialize();
    logEvent("wakeword_init", { status: "ready" });
  } catch (error) {
    logEvent("wakeword_init", {
      status: "error",
      message: error?.message || String(error),
    });
  }
}

initializeOnPageLoad();
