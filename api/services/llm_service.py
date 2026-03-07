from __future__ import annotations


class LlmService:
    def generate(self, transcript: str) -> str:
        if not transcript.strip():
            return "I did not hear a command."
        return f"Dexter stub response: {transcript}."
