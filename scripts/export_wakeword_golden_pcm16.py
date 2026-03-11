#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path("/Users/davidjvitale/workspace/dexter")
REFERENCE_DIR = ROOT / "ai_docs" / "deepcorelabs_openwake_reference_files"
OUT_PATH = ROOT / "frontend" / "wakeword" / "tests" / "wakeword_golden_pcm16.json"
TARGET_SR = 16_000


def read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        nch = wav_file.getnchannels()
        sr = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width={sample_width} for {path}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    return data.astype(np.float32), sr


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if len(audio) == 0:
        return audio

    src_x = np.arange(len(audio), dtype=np.float64)
    dst_len = int(round(len(audio) * (dst_sr / src_sr)))
    if dst_len <= 1:
        return np.zeros((0,), dtype=np.float32)
    dst_x = np.linspace(0, len(audio) - 1, dst_len, dtype=np.float64)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def expected_label(path: Path) -> str:
    name = path.stem.lower()
    if "start" in name:
        return "dexter_start"
    if "stop" in name:
        return "dexter_stop"
    if "abort" in name:
        return "dexter_abort"
    return "unrelated"


def main() -> int:
    wav_files = sorted(REFERENCE_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {REFERENCE_DIR}")

    out_files = []
    for wav_path in wav_files:
        audio, src_sr = read_wav_mono_float32(wav_path)
        audio_16k = resample_linear(audio, src_sr, TARGET_SR)
        pcm16 = float_to_int16(audio_16k)
        payload_b64 = base64.b64encode(pcm16.astype("<i2").tobytes()).decode("ascii")
        out_files.append(
            {
                "name": wav_path.name,
                "expected": expected_label(wav_path),
                "sampleRateInput": src_sr,
                "sampleRateRuntime": TARGET_SR,
                "sampleCount": int(pcm16.size),
                "durationSec16k": round(pcm16.size / TARGET_SR, 4),
                "pcm16LeBase64": payload_b64,
            }
        )

    payload = {
        "format": "wakeword_golden_pcm16_v1",
        "generatedAtEpoch": int(time.time()),
        "files": out_files,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT_PATH} with {len(out_files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
