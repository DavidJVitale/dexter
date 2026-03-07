from __future__ import annotations


class SttService:
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16_000) -> str:
        if not audio_bytes:
            return ""
        seconds = len(audio_bytes) / (sample_rate * 2)
        return f"stub transcript ({seconds:.1f}s of speech)"
