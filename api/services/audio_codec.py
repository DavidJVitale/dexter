from __future__ import annotations

import base64
import io
import math
import struct
import wave


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def synthesize_tone_wav(text: str, sample_rate: int = 16_000) -> bytes:
    duration_sec = min(2.5, max(0.35, 0.35 + (len(text) * 0.015)))
    total_frames = int(sample_rate * duration_sec)
    frequency_hz = 330.0
    amplitude = 0.2

    pcm = bytearray()
    for i in range(total_frames):
        t = i / sample_rate
        sample = amplitude * math.sin(2.0 * math.pi * frequency_hz * t)
        pcm.extend(struct.pack("<h", int(sample * 32767)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))
    return buffer.getvalue()
