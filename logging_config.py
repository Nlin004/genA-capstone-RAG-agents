"""Structured JSON logging shared by every agent, script, the CLI and the API.

Each log record is emitted as one JSON line so it's easy to grep/parse:
query, selected agent, retrieved sources, generated SQL, execution time,
and errors are all passed via the `extra=` dict on the logging call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import get_settings

_RESERVED_FIELDS = set(logging.makeLogRecord({}).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via extra={...} shows up as extra attributes on the
        # record; surface all of it instead of hardcoding field names here.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_FIELDS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name)
