"""Minimal HTTP client for the FastAPI OpenEnv server."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict


DEFAULT_BASE_URL = "http://127.0.0.1:7860"


class WarehouseEnvClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str = "easy", seed: int = 42, session_id: str | None = None) -> Dict[str, Any]:
        payload = {"task_name": task_name, "seed": seed, "session_id": session_id}
        return self._post("/reset", payload)

    def step(self, session_id: str, action_type: int) -> Dict[str, Any]:
        return self._post("/step", {"session_id": session_id, "action_type": action_type})

    def state(self, session_id: str) -> Dict[str, Any]:
        return self._post("/state", {"session_id": session_id})

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))


def run_scenarios(base_url: str = DEFAULT_BASE_URL, config_path: str = "scenario_config.json") -> list[dict]:
    client = WarehouseEnvClient(base_url=base_url)
    scenarios = json.loads(Path(config_path).read_text())
    results: list[dict] = []
    for scenario in scenarios.get("scenarios", []):
        reset_payload = client.reset(
            task_name=scenario["task_name"],
            seed=scenario.get("seed", 42),
        )
        session_id = reset_payload["session_id"]
        done = False
        last_step: Dict[str, Any] | None = None
        for action_type in scenario.get("actions", []):
            last_step = client.step(session_id=session_id, action_type=action_type)
            done = last_step["done"]
            if done:
                break
        state_payload = client.state(session_id=session_id)
        results.append(
            {
                "name": scenario["name"],
                "session_id": session_id,
                "done": done,
                "last_step": last_step,
                "state": state_payload["state"],
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(run_scenarios(), indent=2))
