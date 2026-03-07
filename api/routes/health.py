from __future__ import annotations

from flask import Blueprint

bp = Blueprint("health", __name__)


@bp.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
