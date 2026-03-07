from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    host: str = os.getenv("DEXTER_HOST", "127.0.0.1")
    port: int = int(os.getenv("DEXTER_PORT", "5000"))
    log_dir: str = os.getenv("DEXTER_LOG_DIR", "logs")
    debug: bool = os.getenv("DEXTER_DEBUG", "0") == "1"
    hf_home: str = os.getenv("HF_HOME", "/Users/davidjvitale/.cache/huggingface")
    hf_hub_cache: str = os.getenv(
        "HUGGINGFACE_HUB_CACHE", "/Users/davidjvitale/.cache/huggingface/hub"
    )
