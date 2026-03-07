from __future__ import annotations

import base64
import logging
from uuid import uuid4

from flask import request
from flask_socketio import SocketIO, emit

from api.models.session_state import SessionState
from api.services.assistant_orchestrator import AssistantOrchestrator

LOGGER = logging.getLogger(__name__)

_sessions: dict[str, SessionState] = {}
_orchestrator = AssistantOrchestrator()


def _state_emit(request_id: str | None, value: str) -> None:
    emit("state", {"request_id": request_id, "value": value})


def _error_emit(request_id: str | None, code: str, message: str) -> None:
    emit("error", {"request_id": request_id, "code": code, "message": message})


def _decode_audio_chunk(data: object) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, dict):
        pcm_b64 = data.get("pcm_b64")
        if isinstance(pcm_b64, str) and pcm_b64:
            return base64.b64decode(pcm_b64)
    return b""


def register_socket_events(socketio: SocketIO) -> None:
    @socketio.on("connect")
    def on_connect() -> None:
        sid = request.sid
        _sessions[sid] = SessionState(session_id=sid)
        _state_emit(None, "idle")

    @socketio.on("disconnect")
    def on_disconnect() -> None:
        sid = request.sid
        _sessions.pop(sid, None)

    @socketio.on("start_capture")
    def on_start_capture(data: dict | None) -> None:
        sid = request.sid
        session = _sessions.setdefault(sid, SessionState(session_id=sid))

        request_id = (data or {}).get("request_id") or str(uuid4())
        session.current_request_id = request_id
        session.audio_chunks.clear()
        session.next_seq = 0
        session.state = "listening"

        _state_emit(request_id, "listening")

    @socketio.on("audio_chunk")
    def on_audio_chunk(data: object) -> None:
        sid = request.sid
        session = _sessions.get(sid)
        if not session or session.state != "listening":
            return

        chunk = _decode_audio_chunk(data)
        if not chunk:
            return

        session.audio_chunks.append(chunk)
        session.next_seq += 1

    @socketio.on("abort_capture")
    def on_abort_capture(data: dict | None) -> None:
        sid = request.sid
        session = _sessions.get(sid)
        if not session:
            return

        request_id = (data or {}).get("request_id") or session.current_request_id
        session.audio_chunks.clear()
        session.current_request_id = None
        session.state = "aborted"
        _state_emit(request_id, "aborted")

        session.state = "idle"
        _state_emit(request_id, "idle")

    @socketio.on("stop_capture")
    def on_stop_capture(data: dict | None) -> None:
        sid = request.sid
        session = _sessions.get(sid)
        if not session:
            _error_emit(None, "NO_SESSION", "No session found")
            return

        request_id = (data or {}).get("request_id") or session.current_request_id
        if session.state != "listening":
            _error_emit(request_id, "NOT_LISTENING", "Session is not currently listening")
            return

        audio_bytes = b"".join(session.audio_chunks)
        session.audio_chunks.clear()

        if not audio_bytes:
            _error_emit(request_id, "EMPTY_AUDIO", "No audio captured")
            session.state = "idle"
            _state_emit(request_id, "idle")
            return

        try:
            session.state = "transcribing"
            _state_emit(request_id, "transcribing")
            result = _orchestrator.process(audio_bytes)

            emit("transcript", {"request_id": request_id, "text": result.transcript})

            session.state = "thinking"
            _state_emit(request_id, "thinking")
            emit("response_text", {"request_id": request_id, "text": result.response_text})

            session.state = "speaking"
            _state_emit(request_id, "speaking")
            emit(
                "response_audio",
                {
                    "request_id": request_id,
                    "mime": result.response_audio_mime,
                    "b64": result.response_audio_b64,
                },
            )

            session.state = "idle"
            session.current_request_id = None
            _state_emit(request_id, "idle")
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("pipeline failure")
            _error_emit(request_id, "PIPELINE_FAILED", str(exc))
            session.state = "error"
            _state_emit(request_id, "error")
            session.state = "idle"
            _state_emit(request_id, "idle")
