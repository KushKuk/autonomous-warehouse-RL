"""Hackathon inference entrypoint for the warehouse environment."""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

from openai import OpenAI

from env.environment import WarehouseEnv
from env.models import Action, ActionType


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "autonomous_warehouse_space")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


SYSTEM_PROMPT = """You are an autonomous warehouse robot controller.
Return exactly one action name from this list and nothing else:
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, PICK_ITEM, DROP_ITEM, RECHARGE."""

_HEADING_NAMES = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
_HEADING_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
_ACTION_MAP: Dict[str, ActionType] = {
    "MOVE_FORWARD": ActionType.MOVE_FORWARD,
    "TURN_LEFT": ActionType.TURN_LEFT,
    "TURN_RIGHT": ActionType.TURN_RIGHT,
    "PICK_ITEM": ActionType.PICK_ITEM,
    "DROP_ITEM": ActionType.DROP_ITEM,
    "RECHARGE": ActionType.RECHARGE,
}


def _parse_action(text: str) -> ActionType:
    cleaned = text.strip().upper().replace(" ", "_")
    for key, value in _ACTION_MAP.items():
        if key in cleaned:
            return value
    return ActionType.MOVE_FORWARD


def _compute_direction(robot_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> str:
    dr = target_pos[0] - robot_pos[0]
    dc = target_pos[1] - robot_pos[1]
    parts: List[str] = []
    if dr < -0.5:
        parts.append("NORTH")
    elif dr > 0.5:
        parts.append("SOUTH")
    if dc < -0.5:
        parts.append("WEST")
    elif dc > 0.5:
        parts.append("EAST")
    return "-".join(parts) if parts else "AT_TARGET"


def _extract_local_grid(
    grid: List[List[int]], robot_pos: Tuple[float, float], radius: int = 3
) -> List[List[str]]:
    rr, rc = int(robot_pos[0]), int(robot_pos[1])
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    local: List[List[str]] = []
    for r in range(rr - radius, rr + radius + 1):
        row: List[str] = []
        for c in range(rc - radius, rc + radius + 1):
            if r == rr and c == rc:
                row.append("R")
            elif 0 <= r < rows and 0 <= c < cols:
                row.append(str(grid[r][c]))
            else:
                row.append("X")
        local.append(row)
    return local


def _obs_to_prompt(obs_dict: dict) -> str:
    robot = obs_dict["robot"]
    target = obs_dict["current_target"]
    grid = obs_dict["grid_map"]
    robot_pos = (robot["position"][0], robot["position"][1])
    heading = robot["heading"]
    hdr, hdc = _HEADING_DELTA[heading]
    ahead_r = int(robot_pos[0]) + hdr
    ahead_c = int(robot_pos[1]) + hdc
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    cell_names = {0: "free", 1: "wall", 2: "station", 3: "charger", 4: "obstacle"}
    if 0 <= ahead_r < rows and 0 <= ahead_c < cols:
        cell_ahead = cell_names.get(grid[ahead_r][ahead_c], "unknown")
    else:
        cell_ahead = "out_of_bounds"

    nav = {
        "direction_to_target": (
            _compute_direction(robot_pos, (target[0], target[1]))
            if target is not None
            else "ALL_DELIVERED"
        ),
        "distance": (
            round(
                math.sqrt(
                    (target[0] - robot_pos[0]) ** 2 + (target[1] - robot_pos[1]) ** 2
                ),
                2,
            )
            if target is not None
            else 0.0
        ),
    }

    payload = {
        "robot_position": robot["position"],
        "heading": f"{heading} ({_HEADING_NAMES[heading]})",
        "battery": robot["battery_level"],
        "carrying_item": robot["carrying_item"],
        "carried_item_id": robot["carried_item_id"],
        "current_target": target,
        "navigation": nav,
        "cell_ahead": cell_ahead,
        "items": obs_dict["item_locations"],
        "stations": obs_dict["stations"],
        "nearby_obstacles": obs_dict["nearby_obstacles"],
        "local_grid": _extract_local_grid(grid, robot_pos, radius=3),
        "steps_remaining": obs_dict["steps_remaining"],
    }
    return json.dumps(payload, separators=(",", ":"))


def _infer_action(client: OpenAI, obs_dict: dict) -> ActionType:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _obs_to_prompt(obs_dict)},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    text = response.choices[0].message.content or "MOVE_FORWARD"
    return _parse_action(text)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def run_episode(task_name: str = TASK_NAME) -> int:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = WarehouseEnv(task_name=task_name, seed=42)
    observation = env.reset()
    rewards: List[str] = []
    step_count = 0
    success = False
    last_error = "null"

    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        max_steps = int(MAX_STEPS_OVERRIDE) if MAX_STEPS_OVERRIDE else None
        done = False
        while not done:
            if max_steps is not None and step_count >= max_steps:
                break

            action_type = _infer_action(client, observation.model_dump(mode="json"))
            observation, reward, done, _ = env.step(Action(action_type=action_type))
            step_count += 1
            reward_str = _format_reward(reward.total)
            rewards.append(reward_str)
            last_error = "null"
            print(
                f"[STEP] step={step_count} action={action_type.name} "
                f"reward={reward_str} done={_format_bool(done)} error={last_error}",
                flush=True,
            )

        success = True
        return 0
    except Exception as exc:
        last_error = str(exc).replace("\n", " ")
        return 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        print(
            f"[END] success={_format_bool(success)} steps={step_count} "
            f"rewards={','.join(rewards)}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(run_episode())
