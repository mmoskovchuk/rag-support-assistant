"""Logging configuration with a per-request correlation id.

Each HTTP request gets an id (taken from the X-Request-ID header or generated).
It is stored in a context variable and injected into every log line, so all
messages produced while handling one request can be found with a single grep.
"""

import contextvars
import logging
import logging.config

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level.upper(), "handlers": ["console"]},
            # Third-party libraries are noisy at INFO level.
            "loggers": {
                "httpx": {"level": "WARNING"},
                "chromadb": {"level": "WARNING"},
                "sentence_transformers": {"level": "WARNING"},
            },
        }
    )
