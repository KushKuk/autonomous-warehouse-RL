from __future__ import annotations

import asyncio
import unittest

from server.apis.openenv import get_environment_state, reset_environment, step_environment
from server.apis.system import healthcheck, list_tasks


class DummyRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class OpenEnvApiTests(unittest.TestCase):
    def test_healthcheck(self) -> None:
        response = healthcheck()
        self.assertEqual(response.status, "ok")

    def test_list_tasks(self) -> None:
        response = list_tasks()
        self.assertIn("easy", response["tasks"])

    def test_reset_step_state_flow(self) -> None:
        reset = asyncio.run(reset_environment(DummyRequest({"task_name": "easy", "seed": 42})))
        self.assertTrue(reset.session_id)
        self.assertEqual(reset.info["task_name"], "easy")

        state = asyncio.run(get_environment_state(DummyRequest({"session_id": reset.session_id})))
        self.assertEqual(state.state["task_name"], "easy")

        step = asyncio.run(
            step_environment(DummyRequest({"session_id": reset.session_id, "action_type": 0}))
        )
        self.assertIn("total", step.reward)
        self.assertIsInstance(step.done, bool)

    def test_reset_without_body_uses_defaults(self) -> None:
        reset = asyncio.run(reset_environment(DummyRequest(ValueError("no body"))))
        self.assertTrue(reset.session_id)
        self.assertEqual(reset.info["task_name"], "easy")


if __name__ == "__main__":
    unittest.main()
