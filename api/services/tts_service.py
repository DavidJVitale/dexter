from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .audio_codec import bytes_to_b64, synthesize_tone_wav


class TtsService:
    def __init__(self) -> None:
        self._voice = "Samantha"

    def synthesize(self, text: str) -> tuple[str, str]:
        normalized = (text or "").strip()
        if not normalized:
            normalized = "I did not hear anything."

        aiff_path: Path | None = None
        wav_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_file:
                aiff_path = Path(tmp_file.name)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav_path = Path(tmp_file.name)

            subprocess.run(
                ["say", "-v", self._voice, "-o", str(aiff_path), normalized],
                check=True,
                capture_output=True,
                text=True,
            )

            # Prefer WAV for browser compatibility when afconvert is available.
            try:
                subprocess.run(
                    [
                        "afconvert",
                        "-f",
                        "WAVE",
                        "-d",
                        "LEI16@16000",
                        str(aiff_path),
                        str(wav_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                wav_bytes = wav_path.read_bytes()
                if wav_bytes:
                    return ("audio/wav", bytes_to_b64(wav_bytes))
            except Exception:
                pass

            aiff_bytes = aiff_path.read_bytes()
            if aiff_bytes:
                return ("audio/x-aiff", bytes_to_b64(aiff_bytes))
        except Exception:
            # Fallback keeps the pipeline functional if `say` fails.
            wav_bytes = synthesize_tone_wav(normalized)
            return ("audio/wav", bytes_to_b64(wav_bytes))
        finally:
            if aiff_path and aiff_path.exists():
                aiff_path.unlink(missing_ok=True)
            if wav_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)

        wav_bytes = synthesize_tone_wav(normalized)
        return ("audio/wav", bytes_to_b64(wav_bytes))
