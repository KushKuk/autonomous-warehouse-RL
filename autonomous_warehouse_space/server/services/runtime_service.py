"""Composition root for the server runtime."""

from __future__ import annotations

from server.calendar_environment import CalendarEnvironmentAdapter
from server.database.session_store import SessionStore


session_store = SessionStore()
runtime_service = CalendarEnvironmentAdapter(session_store)
