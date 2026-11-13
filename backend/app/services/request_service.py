"""Service layer for the orchestration request queue.

Submissions land in the queue here; the background worker consumes them.
"""

from __future__ import annotations

from typing import Any

from app.db import connect
from app.repositories import orchestrations as orch_repo
from app.repositories import requests as req_repo
from app.services.orchestrate_service import execute_orchestration

MAX_KEY_LENGTH = 128
MAX_NOTE_LENGTH = 500


class IdempotencyKeyError(ValueError):
    """Raised when an idempotency key is missing or malformed."""


def _validate_key(idempotency_key: str) -> str:
    key = (idempotency_key or "").strip()
    if not key:
        raise IdempotencyKeyError("idempotency_key is required")
    if len(key) > MAX_KEY_LENGTH:
        raise IdempotencyKeyError(
            f"idempotency_key must be at most {MAX_KEY_LENGTH} characters"
        )
    return key


def _start_extra_orchestration(conn, request: dict[str, Any]) -> None:
    """Continue queue processing for an existing idempotency key."""
    orch_id = orch_repo.create_orchestration(conn)
    req_repo.attach_orchestration(
        conn, request_id=request["id"], orchestration_id=orch_id
    )
    execute_orchestration(conn, orch_id)
    req_repo.mark_succeeded(conn, request_id=request["id"])
    conn.commit()


def enqueue_request(idempotency_key: str, note: str | None) -> tuple[dict[str, Any], bool]:
    """Enqueue one orchestration request, deduplicating on the idempotency key."""
    key = _validate_key(idempotency_key)
    clean_note = note.strip() if isinstance(note, str) and note.strip() else None
    if clean_note and len(clean_note) > MAX_NOTE_LENGTH:
        clean_note = clean_note[:MAX_NOTE_LENGTH]
    conn = connect()
    try:
        request, created = req_repo.enqueue(conn, idempotency_key=key, note=clean_note)
        if not created and request is not None:
            _start_extra_orchestration(conn, request)
            request = req_repo.get_request_by_key(conn, key) or request
        return request, created
    finally:
        conn.close()


def list_requests(limit: int = 100) -> list[dict[str, Any]]:
    conn = connect()
    try:
        return req_repo.list_requests(conn, limit=limit)
    finally:
        conn.close()


def get_request(request_id: int) -> dict[str, Any] | None:
    conn = connect()
    try:
        return req_repo.get_request(conn, request_id)
    finally:
        conn.close()


def get_request_report(request_id: int) -> dict[str, Any] | None:
    conn = connect()
    try:
        return req_repo.request_report_link(conn, request_id)
    finally:
        conn.close()


def request_stats() -> dict[str, int]:
    """Queue accounting: enqueue/dedupe/success/failure counters."""
    conn = connect()
    try:
        stats = req_repo.stats(conn)
        stats["linked_orchestrations"] = stats.get("succeeded", 0) + stats.get("failed", 0)
        return stats
    finally:
        conn.close()
