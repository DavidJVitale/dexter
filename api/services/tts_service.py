from __future__ import annotations

from .audio_codec import bytes_to_b64, synthesize_tone_wav


class TtsService:
    def synthesize(self, text: str) -> tuple[str, str]:
        wav_bytes = synthesize_tone_wav(text)
        return ("audio/wav", bytes_to_b64(wav_bytes))
