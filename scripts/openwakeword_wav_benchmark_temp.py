#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
from openwakeword.model import Model

DEXTER_START_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_start.onnx"
DEXTER_STOP_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_stop.onnx"
DEXTER_ABORT_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_abort.onnx"

MELSPEC_MODEL = "/Users/davidjvitale/workspace/dexter/models/openwakeword_resources/melspectrogram.onnx"
EMBEDDING_MODEL = "/Users/davidjvitale/workspace/dexter/models/openwakeword_resources/embedding_model.onnx"

WAVS = [
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_start.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_start2.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_stop.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_stop2.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_abort.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/dexter_abort2.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/unrelated.wav",
    "/Users/davidjvitale/workspace/dexter/ai_docs/deepcorelabs_openwake_reference_files/unrelated2.wav",
]

TARGET_SR = 16_000
CHUNK_SAMPLES = 1280


def _read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        nch = wav.getnchannels()
        sr = wav.getframerate()
        sw = wav.getsampwidth()
        nframes = wav.getnframes()
        raw = wav.readframes(nframes)

    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sw} for {path}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    return data.astype(np.float32), sr


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
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


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _expected_label(path: str) -> str:
    name = Path(path).stem.lower()
    if "start" in name:
        return "dexter_start"
    if "stop" in name:
        return "dexter_stop"
    if "abort" in name:
        return "dexter_abort"
    return "unrelated"


def _empty_scores() -> dict[str, list[float]]:
    return {"dexter_start": [], "dexter_stop": [], "dexter_abort": []}


def _normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    out = {"dexter_start": 0.0, "dexter_stop": 0.0, "dexter_abort": 0.0}
    for key, value in raw.items():
        k = key.lower()
        if "start" in k:
            out["dexter_start"] = float(value)
        elif "stop" in k:
            out["dexter_stop"] = float(value)
        elif "abort" in k:
            out["dexter_abort"] = float(value)
    return out


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"peak": 0.0, "mean": 0.0, "p95": 0.0, "peak_frame": -1, "n": 0}
    arr = np.array(values, dtype=np.float32)
    peak_idx = int(arr.argmax())
    return {
        "peak": float(arr[peak_idx]),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "peak_frame": peak_idx,
        "n": int(arr.size),
    }


def main() -> int:
    for model_path in [DEXTER_START_MODEL, DEXTER_STOP_MODEL, DEXTER_ABORT_MODEL, MELSPEC_MODEL, EMBEDDING_MODEL]:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing required model: {model_path}")

    model = Model(
        wakeword_models=[DEXTER_START_MODEL, DEXTER_STOP_MODEL, DEXTER_ABORT_MODEL],
        inference_framework="onnx",
        melspec_model_path=MELSPEC_MODEL,
        embedding_model_path=EMBEDDING_MODEL,
    )

    started = time.time()
    results: list[dict[str, object]] = []

    for wav_path in WAVS:
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        model.reset()

        audio_f32, src_sr = _read_wav_mono_float32(path)
        audio_16k = _resample_linear(audio_f32, src_sr, TARGET_SR)
        pcm16 = _float_to_int16(audio_16k)

        scores = _empty_scores()

        for i in range(0, len(pcm16), CHUNK_SAMPLES):
            chunk = pcm16[i : i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                pad = np.zeros(CHUNK_SAMPLES - len(chunk), dtype=np.int16)
                chunk = np.concatenate([chunk, pad])

            raw = model.predict(chunk)
            normalized = _normalize_scores(raw)
            for label in scores:
                scores[label].append(normalized[label])

        label_stats = {label: _stats(values) for label, values in scores.items()}
        winner = max(label_stats, key=lambda k: label_stats[k]["peak"])

        file_result = {
            "file": str(path),
            "expected": _expected_label(wav_path),
            "sample_rate_input": src_sr,
            "sample_rate_runtime": TARGET_SR,
            "chunk_samples": CHUNK_SAMPLES,
            "duration_sec_16k": round(len(pcm16) / TARGET_SR, 4),
            "winner_by_peak": winner,
            "scores": label_stats,
        }
        results.append(file_result)

        print(
            f"{path.name}: expected={file_result['expected']} winner={winner} "
            f"start_peak={label_stats['dexter_start']['peak']:.3f} "
            f"stop_peak={label_stats['dexter_stop']['peak']:.3f} "
            f"abort_peak={label_stats['dexter_abort']['peak']:.3f}"
        )

    payload = {
        "benchmark": "openwakeword_wav_benchmark_temp_v1",
        "generated_at_epoch": int(time.time()),
        "elapsed_sec": round(time.time() - started, 3),
        "files": results,
    }

    print("\nBEGIN_BENCHMARK_JSON")
    print(json.dumps(payload, indent=2))
    print("END_BENCHMARK_JSON")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
