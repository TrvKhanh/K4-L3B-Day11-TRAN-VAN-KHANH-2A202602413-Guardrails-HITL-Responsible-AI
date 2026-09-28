"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        req_id = request_id or f"{user_id}_{len(self.logs)}_{len(self._open)}"
        self._open[req_id] = {
            "user_id": user_id,
            "text": text,
            "start_time": datetime.now(timezone.utc),
        }
        return req_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        req_id = request_id or f"{user_id}_{len(self.logs)}"
        open_data = self._open.pop(req_id, {})
        start_time = open_data.get("start_time") or datetime.now(timezone.utc)
        now = datetime.now(timezone.utc)
        latency_ms = (now - start_time).total_seconds() * 1000

        entry = {
            "timestamp": now.isoformat(),
            "request_id": req_id,
            "user_id": user_id,
            "input_text": open_data.get("text", ""),
            "response_text": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 2),
        }
        self.logs.append(entry)

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target = Path(filepath or default_audit_log_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
