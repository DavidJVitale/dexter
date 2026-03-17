from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, Response, send_from_directory
from flask_socketio import SocketIO

from .config import AppConfig
from .routes.health import bp as health_bp


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
    os.environ.setdefault("HF_HOME", config.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", config.hf_hub_cache)

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    models_dir = Path(__file__).resolve().parents[1] / "models"
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

    @app.get("/models/<path:model_path>")
    def model_asset(model_path: str) -> object:
        return send_from_directory(models_dir, model_path)

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    app.register_blueprint(health_bp)
    return app


def create_socketio(app: Flask) -> SocketIO:
    from .routes.socket_events import register_socket_events

    socketio = SocketIO(
        app,
        async_mode="threading",
        cors_allowed_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:5050",
            "http://localhost:5050",
            "http://127.0.0.1:5000",
            "http://localhost:5000",
        ],
        cors_credentials=False,
    )
    register_socket_events(socketio)
    return socketio


app = create_app()
socketio = create_socketio(app)


if __name__ == "__main__":
    config = app.config["APP_CONFIG"]
    os.environ.setdefault("HF_HOME", config.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", config.hf_hub_cache)
    socketio.run(
        app,
        host=config.host,
        port=config.port,
        debug=config.debug,
        allow_unsafe_werkzeug=True,
    )
