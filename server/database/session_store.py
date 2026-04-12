"""SQLite-backed metadata store for runtime sessions and transitions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from server.utils.paths import DATA_DIR, DB_PATH


class SessionStore:
    """Persist coarse-grained session metadata and transition history."""

    def __init__(self, db_path: str | None = None) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or str(DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    done INTEGER NOT NULL,
                    step_count INTEGER NOT NULL,
                    last_state_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    action_type INTEGER,
                    reward_total REAL,
                    done INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def upsert_session(
        self,
        *,
        session_id: str,
        task_name: str,
        seed: int,
        done: bool,
        step_count: int,
        last_state: Dict[str, Any],
    ) -> None:
        payload = json.dumps(last_state)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, task_name, seed, done, step_count, last_state_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    task_name=excluded.task_name,
                    seed=excluded.seed,
                    done=excluded.done,
                    step_count=excluded.step_count,
                    last_state_json=excluded.last_state_json
                """,
                (session_id, task_name, seed, int(done), step_count, payload),
            )

    def record_transition(
        self,
        *,
        session_id: str,
        step_count: int,
        action_type: Optional[int],
        reward_total: Optional[float],
        done: bool,
        payload: Dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transitions(
                    session_id, step_count, action_type, reward_total, done, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    step_count,
                    action_type,
                    reward_total,
                    int(done),
                    json.dumps(payload),
                ),
            )

    def fetch_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, task_name, seed, done, step_count, last_state_json
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "task_name": row[1],
            "seed": row[2],
            "done": bool(row[3]),
            "step_count": row[4],
            "last_state": json.loads(row[5]),
        }

    def fetch_latest_session(self) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, task_name, seed, done, step_count, last_state_json
                FROM sessions
                ORDER BY rowid DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "task_name": row[1],
            "seed": row[2],
            "done": bool(row[3]),
            "step_count": row[4],
            "last_state": json.loads(row[5]),
        }
