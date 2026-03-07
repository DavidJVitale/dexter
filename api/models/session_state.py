from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    state: str = "idle"
    current_request_id: str | None = None
    audio_chunks: list[bytes] = field(default_factory=list)
    next_seq: int = 0
