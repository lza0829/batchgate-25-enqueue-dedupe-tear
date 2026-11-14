"""Service layer for the orchestration request queue.

Submissions land in the queue here; the background worker consumes them.
"""

from __future__ import annotations

from app.db import connect
from app.repositories import requests as req_repo

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


def enqueue_request(idempotency_key: str, note: str | None) -> tuple[dict[str, Any], bool]:
    """Enqueue one orchestration request, deduplicating on the idempotency key.

    A repeated key only returns the existing request (with ``dedupe_count``
    incremented inside the repository transaction). It never starts an
    orchestration and never alters the existing request's status, linked
    orchestration, error or timestamps -- so a request that already
    succeeded or failed is left in its terminal state.
    """
    key = _validate_key(idempotency_key)
    clean_note = note.strip() if isinstance(note, str) and note.strip() else None
    if clean_note and len(clean_note) > MAX_NOTE_LENGTH:
        clean_note = clean_note[:MAX_NOTE_LENGTH]
    conn = connect()
    try:
        return req_repo.enqueue(conn, idempotency_key=key, note=clean_note)
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
    """Queue accounting: enqueue/dedupe/success/failure counters.

    ``linked_orchestrations`` is the actual number of request rows linked to
    an orchestration record; with one run per consumed request it equals
    succeeded + failed, and ``shadow_runs`` (orchestrations no request points
    at) stays zero -- the request list, counters and runs reconcile 1:1.
    """
    conn = connect()
    try:
        return req_repo.stats(conn)
    finally:
        conn.close()
