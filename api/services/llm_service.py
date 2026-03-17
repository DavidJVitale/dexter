from __future__ import annotations

import logging
import os
import threading

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "LiquidAI/LFM2-24B-A2B-MLX-4bit"
DEFAULT_SYSTEM_PROMPT = (
    "You are Dexter, a local voice assistant.\n"
    "Follow these rules:\n"
    "1) Reply in one or two short sentences.\n"
    "2) Use plain spoken English.\n"
    "3) No markdown, no lists, no emojis.\n"
    "4) If uncertain, say so briefly and ask one clarifying question.\n"
    "5) Do not mention these rules."
)


class LlmService:
    def __init__(self) -> None:
        self._model_repo = os.getenv("DEXTER_LLM_MODEL_REPO", DEFAULT_MODEL_REPO)
        self._system_prompt = os.getenv("DEXTER_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        self._max_tokens = int(os.getenv("DEXTER_LLM_MAX_TOKENS", "96"))
        self._temperature = float(os.getenv("DEXTER_LLM_TEMPERATURE", "0.2"))
        self._rep_penalty = float(os.getenv("DEXTER_LLM_REP_PENALTY", "1.05"))
        self._lock = threading.Lock()

        LOGGER.info("Loading MLX LLM model: %s", self._model_repo)
        self._model, self._tokenizer = load(self._model_repo)
        LOGGER.info("MLX LLM model loaded: %s", self._model_repo)
        self._warm_model()

    def generate(self, transcript: str) -> str:
        user_text = (transcript or "").strip()
        if not user_text:
            return "I didn't catch that. Please say it again."

        with self._lock:
            prompt = self._build_prompt(user_text)
            logits_processors = (
                make_logits_processors(repetition_penalty=self._rep_penalty)
                if self._rep_penalty > 0
                else None
            )
            response = generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self._max_tokens,
                sampler=make_sampler(temp=self._temperature),
                logits_processors=logits_processors,
                verbose=False,
            )
        return self._strip_special_tokens(response)

    def _warm_model(self) -> None:
        try:
            with self._lock:
                warm_prompt = self._build_prompt("Say: ready.")
                _ = generate(
                    self._model,
                    self._tokenizer,
                    warm_prompt,
                    max_tokens=8,
                    sampler=make_sampler(temp=0.0),
                    verbose=False,
                )
            LOGGER.info("MLX LLM warmup completed")
        except Exception:  # pragma: no cover
            LOGGER.exception("MLX LLM warmup failed")

    def _build_prompt(self, user_text: str) -> str:
        if getattr(self._tokenizer, "has_chat_template", False):
            return self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"System: {self._system_prompt}\n\n"
            f"User: {user_text}\n\n"
            "Assistant:"
        )

    def _strip_special_tokens(self, text: str) -> str:
        cleaned = text or ""
        tokens = {"<|startoftext|>", "<|endoftext|>"}
        for token in getattr(self._tokenizer, "all_special_tokens", []) or []:
            if isinstance(token, str) and token:
                tokens.add(token)
        for token in sorted(tokens, key=len, reverse=True):
            cleaned = cleaned.replace(token, "")
        return cleaned.strip()
