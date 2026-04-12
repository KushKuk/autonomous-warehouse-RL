"""Compatibility wrapper exposing warehouse env reset/step/state semantics for HTTP."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict

from env.environment import WarehouseEnv
from env.models import Action, ActionType
from env.tasks import TASKS
from server.database.session_store import SessionStore


@dataclass
class ActiveSession:
    env: WarehouseEnv
    task_name: str
    seed: int


class CalendarEnvironmentAdapter:
    """
    Server-side OpenEnv adapter.

    The file name follows the reference layout requested in `summary.txt`,
    but the wrapped domain remains the warehouse environment.
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._sessions: Dict[str, ActiveSession] = {}
        self._lock = threading.Lock()

    def list_tasks(self) -> list[str]:
        return list(TASKS.keys())

    def reset(self, *, task_name: str, seed: int, session_id: str | None = None) -> dict:
        session_id = session_id or str(uuid.uuid4())
        env = WarehouseEnv(task_name=task_name, seed=seed)
        observation = env.reset(seed=seed).model_dump(mode="json")
        state = env.state().model_dump(mode="json")
        payload = {
            "session_id": session_id,
            "observation": observation,
            "info": {
                "task_name": task_name,
                "seed": seed,
                "available_tasks": self.list_tasks(),
            },
        }
        with self._lock:
            self._sessions[session_id] = ActiveSession(env=env, task_name=task_name, seed=seed)
        self._store.upsert_session(
            session_id=session_id,
            task_name=task_name,
            seed=seed,
            done=state["done"],
            step_count=state["observation"]["episode_time"],
            last_state=state,
        )
        self._store.record_transition(
            session_id=session_id,
            step_count=0,
            action_type=None,
            reward_total=None,
            done=False,
            payload=payload,
        )
        return payload

    def step(self, *, session_id: str, action_type: int) -> dict:
        session = self._get_session(session_id)
        action = Action(action_type=ActionType(action_type))
        observation, reward, done, info = session.env.step(action)
        grade = session.env.grade() if done else None
        state = session.env.state().model_dump(mode="json")
        payload = {
            "session_id": session_id,
            "observation": observation.model_dump(mode="json"),
            "reward": reward.model_dump(mode="json"),
            "done": done,
            "info": info,
            "grade": grade,
        }
        self._store.upsert_session(
            session_id=session_id,
            task_name=session.task_name,
            seed=session.seed,
            done=done,
            step_count=info["step"],
            last_state=state,
        )
        self._store.record_transition(
            session_id=session_id,
            step_count=info["step"],
            action_type=action_type,
            reward_total=reward.total,
            done=done,
            payload=payload,
        )
        return payload

    def state(self, *, session_id: str) -> dict:
        session = self._get_session(session_id)
        state = session.env.state().model_dump(mode="json")
        self._store.upsert_session(
            session_id=session_id,
            task_name=session.task_name,
            seed=session.seed,
            done=state["done"],
            step_count=state["observation"]["episode_time"],
            last_state=state,
        )
        return {
            "session_id": session_id,
            "state": state,
        }

    def _get_session(self, session_id: str) -> ActiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id '{session_id}'. Call /reset first.")
        return session
