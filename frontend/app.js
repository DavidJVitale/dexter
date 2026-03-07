import { AudioCapture } from "./audio_capture.js";
import { createSocketClient } from "./socket_client.js";
import { OpenWakeWordAdapter } from "./wakeword_openwakeword_adapter.js";

const statePill = document.getElementById("state-pill");
const requestIdEl = document.getElementById("request-id");
const transcriptEl = document.getElementById("transcript");
const responseTextEl = document.getElementById("response-text");
const responseAudioEl = document.getElementById("response-audio");
const eventLogEl = document.getElementById("event-log");

const enableMicButton = document.getElementById("enable-mic");
const startButton = document.getElementById("start-capture");
const stopButton = document.getElementById("stop-capture");
const abortButton = document.getElementById("abort-capture");

let currentRequestId = null;

function makeRequestId() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `req_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}

function logEvent(type, payload) {
  const line = `${new Date().toISOString()} ${type} ${JSON.stringify(payload)}`;
  eventLogEl.textContent = `${line}\n${eventLogEl.textContent}`;
}

function setState(state) {
  statePill.textContent = state;
  statePill.className = `state ${state}`;
}

const socket = createSocketClient();
const wakeword = new OpenWakeWordAdapter();
const audioCapture = new AudioCapture({
  onChunk: (chunkPayload) => {
    if (!currentRequestId) {
      return;
    }
    socket.emit("audio_chunk", { ...chunkPayload, request_id: currentRequestId });
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
  responseTextEl.textContent = payload.text || "(none)";
  logEvent("response_text", payload);
});

socket.on("response_audio", (payload) => {
  responseAudioEl.src = `data:${payload.mime};base64,${payload.b64}`;
  responseAudioEl.play().catch(() => {});
  logEvent("response_audio", { request_id: payload.request_id, mime: payload.mime });
});

socket.on("error", (payload) => {
  logEvent("error", payload);
});

enableMicButton.addEventListener("click", async () => {
  await wakeword.initialize();
  await audioCapture.initialize();
  logEvent("mic", { status: "enabled" });
});

startButton.addEventListener("click", () => {
  currentRequestId = makeRequestId();
  requestIdEl.textContent = currentRequestId;
  transcriptEl.textContent = "(none)";
  responseTextEl.textContent = "(none)";

  audioCapture.startCapture();
  socket.emit("start_capture", { session_id: socket.id, request_id: currentRequestId });
  logEvent("start_capture", { request_id: currentRequestId });
});

stopButton.addEventListener("click", () => {
  audioCapture.stopCapture();
  socket.emit("stop_capture", { request_id: currentRequestId });
  logEvent("stop_capture", { request_id: currentRequestId });
});

abortButton.addEventListener("click", () => {
  audioCapture.stopCapture();
  socket.emit("abort_capture", { request_id: currentRequestId, reason: "manual_abort_debug" });
  logEvent("abort_capture", { request_id: currentRequestId });
});
