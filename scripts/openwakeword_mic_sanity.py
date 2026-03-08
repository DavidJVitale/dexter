#!/usr/bin/env python3
from __future__ import annotations

import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import download_models

# Hardcoded paths for quick sanity testing.
# Replace these with your actual file paths.
DEXTER_START_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_start.onnx"
DEXTER_STOP_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_stop.onnx"
DEXTER_ABORT_MODEL = "/Users/davidjvitale/workspace/dexter/models/wakewords/dexter_abort.onnx"
OWW_RESOURCES_DIR = "/Users/davidjvitale/workspace/dexter/models/openwakeword_resources"

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 1280
HIT_THRESHOLD = 0.6
HIT_COOLDOWN_SECONDS = 1.0


def _validate_paths(paths: list[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print("Missing model files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)


def _ensure_openwakeword_resources(resources_dir: str) -> tuple[str, str]:
    resource_path = Path(resources_dir)
    resource_path.mkdir(parents=True, exist_ok=True)

    melspec_path = resource_path / "melspectrogram.onnx"
    embedding_path = resource_path / "embedding_model.onnx"

    if not melspec_path.exists() or not embedding_path.exists():
        print("Bootstrapping openWakeWord feature models into local resources dir...")
        download_models(target_directory=str(resource_path))

    if not melspec_path.exists() or not embedding_path.exists():
        print("Failed to prepare openWakeWord feature models:", file=sys.stderr)
        print(f"  expected: {melspec_path}", file=sys.stderr)
        print(f"  expected: {embedding_path}", file=sys.stderr)
        sys.exit(1)

    return str(melspec_path), str(embedding_path)


def _norm_scores(raw: dict[str, float]) -> dict[str, float]:
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


def main() -> int:
    model_paths = [DEXTER_START_MODEL, DEXTER_STOP_MODEL, DEXTER_ABORT_MODEL]
    _validate_paths(model_paths)
    melspec_model_path, embedding_model_path = _ensure_openwakeword_resources(OWW_RESOURCES_DIR)

    model = Model(
        wakeword_models=model_paths,
        inference_framework="onnx",
        melspec_model_path=melspec_model_path,
        embedding_model_path=embedding_model_path,
    )

    audio_q: queue.Queue[bytes] = queue.Queue(maxsize=100)
    running = True

    def _sigint_handler(_sig: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    def callback(indata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        del frames
        if status:
            print(f"\nAudio callback status: {status}", file=sys.stderr)
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    print("Listening... Press Ctrl+C to stop.")
    print(f"sample_rate={SAMPLE_RATE}, chunk_samples={CHUNK_SAMPLES}")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        last_hit_ts = {
            "dexter_start": 0.0,
            "dexter_stop": 0.0,
            "dexter_abort": 0.0,
        }
        while running:
            try:
                chunk_bytes = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            raw_scores = model.predict(chunk)
            scores = _norm_scores(raw_scores)

            line = (
                f"dexter_start={scores['dexter_start']:.3f}  "
                f"dexter_stop={scores['dexter_stop']:.3f}  "
                f"dexter_abort={scores['dexter_abort']:.3f}"
            )
            print(f"\r{line}", end="", flush=True)

            now = time.monotonic()
            for label, score in scores.items():
                if score <= HIT_THRESHOLD:
                    continue
                if (now - last_hit_ts[label]) < HIT_COOLDOWN_SECONDS:
                    continue
                last_hit_ts[label] = now
                print(f"\nHIT {label}: {score:.3f}", flush=True)
            time.sleep(0.001)

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
