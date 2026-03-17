from __future__ import annotations

import os
import numpy as np


class SttService:
    def __init__(self) -> None:
        self._model_repo = os.getenv("DEXTER_STT_MODEL_REPO", "mlx-community/whisper-medium-mlx")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16_000) -> str:
        if not audio_bytes:
            return ""

        try:
            import mlx_whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mlx-whisper is not installed. Install it in dexter/.venv to enable STT."
            ) from exc

        # Convert incoming mono PCM16 bytes to float32 waveform in [-1, 1].
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm16.size == 0:
            return ""
        audio_f32 = pcm16.astype(np.float32) / 32768.0

        result = mlx_whisper.transcribe(
            audio_f32,
            path_or_hf_repo=self._model_repo,
        )
        text = str((result or {}).get("text", "")).strip()
        return text
