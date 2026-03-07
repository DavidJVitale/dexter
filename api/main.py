from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, send_from_directory
from flask_socketio import SocketIO

from .config import AppConfig
from .routes.health import bp as health_bp
from .routes.socket_events import register_socket_events


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(),
        ],
    )


def create_app() -> Flask:
    config = AppConfig()
    _configure_logging(Path(config.log_dir))

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    app = Flask(
        __name__,
        static_folder=str(frontend_dir),
        static_url_path="",
    )
    app.config["APP_CONFIG"] = config

    @app.get("/")
    def index() -> object:
        return send_from_directory(frontend_dir, "index.html")

    @app.get("/frontend/<path:asset_path>")
    def frontend_asset(asset_path: str) -> object:
        return send_from_directory(frontend_dir, asset_path)

    app.register_blueprint(health_bp)
    return app


def create_socketio(app: Flask) -> SocketIO:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    register_socket_events(socketio)
    return socketio


app = create_app()
socketio = create_socketio(app)


if __name__ == "__main__":
    config = app.config["APP_CONFIG"]
    os.environ.setdefault("HF_HOME", config.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", config.hf_hub_cache)
    socketio.run(app, host=config.host, port=config.port, debug=config.debug)
