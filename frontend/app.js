import { AudioCapture } from "./audio_capture.js";
import { createSocketClient } from "./socket_client.js";
import { OpenWakeWordAdapter } from "./wakeword_openwakeword_adapter.js";

const screenRoot = document.getElementById("screen-root");
const engineeringShell = document.getElementById("engineering-shell");
const sleekShell = document.getElementById("sleek-shell");
const sleekOrb = document.getElementById("sleek-orb");
const sleekIcon = document.getElementById("sleek-icon");
const sleekCaption = document.getElementById("sleek-caption");
const modeMenuButton = document.getElementById("mode-menu-button");
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
let currentVisualMode = "engineering";
let visualStateOverride = null;
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
const VISUAL_MODES = {
  engineering: {
    background: "linear-gradient(135deg, #f6f8fb, #e9eff8)",
    orb: "#f4f5f7",
    icon: "#111111",
  },
  sleek_waiting: {
    background: "#f2f2ef",
    orb: "#f8f8f6",
    icon: "#111111",
  },
  sleek_listening: {
    background: "#fff6e7",
    orb: "#bb6a11",
    icon: "#ffffff",
  },
  sleek_thinking: {
    background: "#dfeaff",
    orb: "#1f5bff",
    icon: "#ffffff",
  },
  sleek_speaking: {
    background: "#e8f6df",
    orb: "#b7e48a",
    icon: "#111111",
  },
};

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

function setBackgroundColor(color) {
  if (!screenRoot) {
    return;
  }
  screenRoot.style.background = color;
}

function setIconColor(color) {
  if (!sleekIcon) {
    return;
  }
  sleekIcon.style.color = color;
}

function setCaptionColor(color) {
  if (!sleekCaption) {
    return;
  }
  sleekCaption.style.color = color;
}

function setOrbColor(color) {
  if (!sleekOrb) {
    return;
  }
  sleekOrb.style.backgroundColor = color;
}

function setVisualMode(modeName) {
  const mode = VISUAL_MODES[modeName];
  if (!mode) {
    return;
  }
  setBackgroundColor(mode.background);
  setOrbColor(mode.orb);
  setIconColor(mode.icon);
  setCaptionColor("#111111");
}

function getActiveVisualState() {
  if (visualStateOverride) {
    return visualStateOverride;
  }
  return mapStateToVisualState(currentState);
}

function applyShellMode(modeName) {
  currentVisualMode = modeName;
  const isSleek = modeName === "sleek";
  engineeringShell.hidden = isSleek;
  sleekShell.hidden = !isSleek;
  setVisualMode(isSleek ? getActiveVisualState() : "engineering");
}

function mapStateToVisualState(state) {
  if (currentVisualMode !== "sleek") {
    return "engineering";
  }
  if (state === "listening") {
    return "sleek_listening";
  }
  if (state === "transcribing" || state === "thinking") {
    return "sleek_thinking";
  }
  if (state === "speaking") {
    return "sleek_speaking";
  }
  return "sleek_waiting";
}

function clearScreenText() {
  requestIdEl.textContent = "";
  transcriptEl.textContent = "";
  responseTextEl.textContent = "";
  latestResponseText = "";
  if (sleekCaption) {
    sleekCaption.textContent = "";
  }
  responseAudioEl.removeAttribute("src");
  eventLogEl.textContent = "";
}

function setState(state) {
  currentState = state;
  statePill.textContent = state;
  statePill.className = `state ${state}`;
  setVisualMode(getActiveVisualState());
}

function beginCapture(trigger = "manual") {
  if (currentState === "listening") {
    return;
  }
  currentRequestId = makeRequestId();
  clearScreenText();
  requestIdEl.textContent = currentRequestId;
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

async function loadSleekIcon() {
  const response = await fetch("./glasses.svg");
  const svgMarkup = await response.text();
  sleekIcon.innerHTML = svgMarkup
    .replace(/fill=\"#000\"/g, 'fill="currentColor"')
    .replace(/fill=\"#000000\"/g, 'fill="currentColor"')
    .replace(/stroke=\"#000\"/g, 'stroke="currentColor"')
    .replace(/stroke=\"#000000\"/g, 'stroke="currentColor"');
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
  if (sleekCaption) {
    sleekCaption.textContent = latestResponseText;
  }
  logEvent("response_text", payload);
});

socket.on("response_audio", (payload) => {
  responseAudioEl.src = `data:${payload.mime};base64,${payload.b64}`;
  visualStateOverride = "sleek_speaking";
  setVisualMode(getActiveVisualState());
  responseAudioEl.play().catch((error) => {
    logEvent("response_audio_play_error", {
      request_id: payload.request_id,
      mime: payload.mime,
      message: error?.message || String(error),
    });
    visualStateOverride = null;
    setVisualMode(getActiveVisualState());
    speakFallbackText(latestResponseText);
  });
  logEvent("response_audio", { request_id: payload.request_id, mime: payload.mime });
});

responseAudioEl.addEventListener("ended", () => {
  visualStateOverride = null;
  setVisualMode(getActiveVisualState());
});

responseAudioEl.addEventListener("pause", () => {
  if (responseAudioEl.ended) {
    return;
  }
  visualStateOverride = null;
  setVisualMode(getActiveVisualState());
});

socket.on("error", (payload) => {
  logEvent("error", payload);
});

enableMicButton.addEventListener("click", async () => {
  await audioCapture.initialize();
  await wakeword.start();
  logEvent("mic", { status: "enabled" });
});

modeMenuButton.addEventListener("click", () => {
  applyShellMode(currentVisualMode === "engineering" ? "sleek" : "engineering");
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
    await loadSleekIcon();
    applyShellMode("engineering");
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
