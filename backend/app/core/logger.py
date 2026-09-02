"""Structured logger for Railway deployment — JSON to stdout, request_id, stack traces."""
import logging, sys, json, uuid, time, traceback
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
LOG_LEVEL = logging.INFO

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))[:2000]
        if hasattr(record, "extra"): payload.update(record.extra)
        return json.dumps(payload)

def setup_logging():
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(LOG_LEVEL)
    for n in ("uvicorn","uvicorn.error","uvicorn.access","sqlalchemy"): logging.getLogger(n).handlers = [h]; logging.getLogger(n).propagate = False

def get_logger(name: str): return logging.getLogger(name)
