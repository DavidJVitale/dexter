from __future__ import annotations

from dataclasses import dataclass

from .llm_service import LlmService
from .stt_service import SttService
from .tts_service import TtsService


@dataclass
class AssistantResult:
    transcript: str
    response_text: str
    response_audio_mime: str
    response_audio_b64: str
    tool_traces: list[dict]


class AssistantOrchestrator:
    def __init__(self) -> None:
        self._stt = SttService()
        self._llm = LlmService()
        self._tts = TtsService()

    def process(self, audio_bytes: bytes) -> AssistantResult:
        transcript = self._stt.transcribe(audio_bytes)
        response_text, tool_traces = self._llm.generate(transcript)
        mime, b64_audio = self._tts.synthesize(response_text)
        return AssistantResult(
            transcript=transcript,
            response_text=response_text,
            response_audio_mime=mime,
            response_audio_b64=b64_audio,
            tool_traces=tool_traces,
        )
