"""
Baseline inference script for the Autonomous Warehouse Robot environment.

Uses the Groq API (OpenAI-compatible) to run an LLM-guided agent on all three
task difficulties (easy, medium, hard) and prints per-task scores.

Usage
-----
    GROQ_API_KEY=<your-key> python baseline/run_agent.py

Environment variables (also loaded from .env file)
---------------------------------------------------
    GROQ_API_KEY     : required - your Groq API key
    OPENAI_MODEL     : optional - model name (default: openai/gpt-oss-120b)
    OPENAI_BASE_URL  : optional - base API URL (default: https://api.groq.com/openai/v1)
    NUM_EPISODES     : optional - episodes per task (default: 3)
    SEED             : optional - base random seed (default: 42)
"""

from __future__ import annotations

import json
import math
import os
import sys
import random
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# Auto-load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # .env loading is optional – use shell exports instead

from env.environment import WarehouseEnv
from env.models import Action, ActionType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
NUM_EPISODES    = int(os.getenv("NUM_EPISODES", "3"))
BASE_SEED       = int(os.getenv("SEED", "42"))
TASK_NAMES      = ["easy", "medium", "hard"]
MAX_TOKENS      = 256


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

# [FIX #6] System prompt now includes action preconditions, grid encoding,
#          heading reference, and strategy that references feedback.
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous warehouse robot controller.
    You receive structured JSON observations and feedback after each action.
    You must output ONLY a single action name — nothing else.

    ACTIONS AND PRECONDITIONS:
      MOVE_FORWARD  – Move one cell in the current heading direction.
                      Fails if the cell ahead is a wall (1), obstacle (4), or out of bounds.
      TURN_LEFT     – Rotate 90° counter-clockwise. Always succeeds.
      TURN_RIGHT    – Rotate 90° clockwise. Always succeeds.
      PICK_ITEM     – Pick up an item. Succeeds ONLY if you are on the EXACT same cell
                      as an unpicked item AND you are NOT already carrying an item.
      DROP_ITEM     – Drop your carried item. Succeeds ONLY if you are carrying an item
                      AND you are standing on that item's CORRECT target delivery station.
      RECHARGE      – Recharge battery. Succeeds ONLY if on a charging station (cell = 3).

    GRID CELL ENCODING (shown in local_grid, R = you):
      0 = free floor       1 = wall/shelf (BLOCKED)
      2 = delivery station  3 = charging station
      4 = dynamic obstacle (BLOCKED, moves periodically)

    HEADING:
      0 = North (row decreases)  1 = East (column increases)
      2 = South (row increases)  3 = West (column decreases)

    STRATEGY:
    1. Read the navigation hints — they tell you which direction the target is
       and what action to take next.
    2. Check "cell_ahead" before using MOVE_FORWARD. If it says WALL or OBSTACLE,
       you MUST turn first.
    3. Use [FEEDBACK] messages to verify your actions succeeded. If you see
       COLLISION or INVALID_ACTION, change your approach immediately.
    4. Each item has a "target_station" — after picking it up, navigate to that
       specific station to drop it.
    5. If battery < 20%, navigate to a charging station and RECHARGE.
    6. If you notice you are repeating the same actions, try something different.

    Output ONLY one action name. No explanation, no punctuation.
""")


# ---------------------------------------------------------------------------
# Heading & navigation helpers
# ---------------------------------------------------------------------------

_HEADING_NAMES = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
_HEADING_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # North
    1:  (0,  1),   # East
    2:  (1,  0),   # South
    3:  (0, -1),   # West
}


# [FIX #3] Compute human-readable direction from robot to target
def _compute_direction(
    robot_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
) -> str:
    """Return cardinal direction string (e.g. 'NORTH-EAST') from robot to target."""
    dr = target_pos[0] - robot_pos[0]   # positive → south
    dc = target_pos[1] - robot_pos[1]   # positive → east

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


# [FIX #3] Suggest the immediate best action to make progress toward the target
def _suggest_action(
    robot_pos: Tuple[float, float],
    robot_heading: int,
    target_pos: Tuple[float, float],
    grid: List[List[int]],
) -> str:
    """Suggest MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT to approach the target."""
    rr, rc = int(robot_pos[0]), int(robot_pos[1])
    tr, tc = int(target_pos[0]), int(target_pos[1])

    dr = tr - rr
    dc = tc - rc

    if dr == 0 and dc == 0:
        return "AT_TARGET"

    # Ideal heading: prefer the axis with the larger remaining distance
    if abs(dr) >= abs(dc):
        ideal_heading = 2 if dr > 0 else 0   # South or North
    else:
        ideal_heading = 1 if dc > 0 else 3   # East or West

    # Already facing the right way?
    if robot_heading == ideal_heading:
        hdr, hdc = _HEADING_DELTA[robot_heading]
        nr, nc = rr + hdr, rc + hdc
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] not in (1, 4):
            return "MOVE_FORWARD"
        # Blocked ahead — suggest a turn to try the secondary axis
        return "TURN_RIGHT"

    # Need to turn
    diff = (ideal_heading - robot_heading) % 4
    if diff == 1:
        return "TURN_RIGHT"
    elif diff == 3:
        return "TURN_LEFT"
    else:
        return "TURN_RIGHT"  # 180° — pick one


# [FIX #2] Extract a local grid patch around the robot so the LLM has spatial
#          awareness without flooding the context with the entire map.
def _extract_local_grid(
    grid: List[List[int]],
    robot_pos: Tuple[float, float],
    radius: int = 3,
) -> List[List[str]]:
    """Return a (2*radius+1)² patch centred on the robot, with 'R' at the centre."""
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
                row.append("X")   # out of bounds
        local.append(row)
    return local


# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs_dict: dict) -> str:
    """Convert observation dict to a rich, context-full LLM prompt."""
    robot   = obs_dict["robot"]
    target  = obs_dict["current_target"]
    grid    = obs_dict["grid_map"]

    robot_pos = (robot["position"][0], robot["position"][1])
    heading   = robot["heading"]

    # [FIX #5] Include target_station in item data
    items = [
        {
            "id":             it["item_id"],
            "pos":            it["position"],
            "picked":         it["is_picked"],
            "target_station": it["target_station"],
        }
        for it in obs_dict["item_locations"]
    ]

    nearby = [
        {"pos": ob["position"], "dynamic": ob["is_dynamic"]}
        for ob in obs_dict["nearby_obstacles"]
    ]

    stations = [
        {"id": st["station_id"], "pos": st["position"]}
        for st in obs_dict["stations"]
    ]

    # [FIX #2] Local grid view
    local_grid = _extract_local_grid(grid, robot_pos, radius=3)

    # [FIX #3] Navigation hints
    nav_hints: Dict[str, object] = {}
    if target is not None:
        target_t = (target[0], target[1])
        nav_hints["direction_to_target"] = _compute_direction(robot_pos, target_t)
        nav_hints["suggested_action"]    = _suggest_action(
            robot_pos, heading, target_t, grid
        )
        nav_hints["distance"] = round(
            math.sqrt(
                (target[0] - robot_pos[0]) ** 2
                + (target[1] - robot_pos[1]) ** 2
            ),
            1,
        )
    else:
        # [FIX #7] Graceful handling when all items are delivered
        nav_hints["direction_to_target"] = "ALL_DELIVERED"
        nav_hints["suggested_action"]    = "NONE"
        nav_hints["distance"]            = 0

    # [FIX #9] Tell the LLM what is directly ahead so it can avoid blind moves
    hdr, hdc = _HEADING_DELTA[heading]
    ahead_r = int(robot_pos[0]) + hdr
    ahead_c = int(robot_pos[1]) + hdc
    grid_rows = len(grid)
    grid_cols = len(grid[0]) if grid_rows else 0
    if 0 <= ahead_r < grid_rows and 0 <= ahead_c < grid_cols:
        _cell_names = {
            0: "free", 1: "WALL", 2: "station", 3: "charger", 4: "OBSTACLE",
        }
        cell_ahead_desc = _cell_names.get(grid[ahead_r][ahead_c], "unknown")
    else:
        cell_ahead_desc = "OUT_OF_BOUNDS"

    summary = {
        "robot_position":    list(robot_pos),
        "robot_heading":     f"{heading} ({_HEADING_NAMES[heading]})",
        "cell_ahead":        cell_ahead_desc,
        "battery":           robot["battery_level"],
        "carrying_item":     robot["carrying_item"],
        "carried_item_id":   robot["carried_item_id"],
        "current_target":    target,
        "navigation":        nav_hints,
        "items":             items,
        "delivery_stations": stations,
        "nearby_obstacles":  nearby,
        "local_grid":        local_grid,
        "steps_remaining":   obs_dict["steps_remaining"],
    }
    return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# Feedback builder (FIX #1 — the most critical fix)
# ---------------------------------------------------------------------------

def _build_feedback(reward, info: dict, action_type: ActionType) -> str:
    """
    Build a concise feedback string that tells the LLM what happened
    as a result of its last action.
    """
    parts: List[str] = [f"Action {action_type.name}"]

    # Reward signal
    parts.append(f"reward={reward.total:+.3f}")

    # Only mention non-zero reward components for brevity
    details: List[str] = []
    if reward.collision_penalty != 0:
        details.append(f"COLLISION({reward.collision_penalty:+.1f})")
    if reward.invalid_action_penalty != 0:
        details.append(f"INVALID_ACTION({reward.invalid_action_penalty:+.1f})")
    if reward.pickup_reward != 0:
        details.append(f"PICKUP_SUCCESS(+{reward.pickup_reward:.1f})")
    if reward.delivery_reward != 0:
        details.append(f"DELIVERY_SUCCESS(+{reward.delivery_reward:.1f})")
    if reward.approach_reward > 0.001:
        details.append("moved_closer")
    elif reward.approach_reward < -0.001:
        details.append("moved_AWAY")
    if reward.battery_penalty != 0:
        details.append("LOW_BATTERY_WARNING")
    if reward.efficiency_bonus != 0:
        details.append(f"EFFICIENCY_BONUS(+{reward.efficiency_bonus:.1f})")

    if details:
        parts.append(" | ".join(details))

    # Progress snapshot
    parts.append(
        f"delivered={info['delivered_items']}/{info['total_items']} "
        f"battery={info['battery_level']:.1f}%"
    )

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_MAP: Dict[str, ActionType] = {
    "MOVE_FORWARD": ActionType.MOVE_FORWARD,
    "TURN_LEFT":    ActionType.TURN_LEFT,
    "TURN_RIGHT":   ActionType.TURN_RIGHT,
    "PICK_ITEM":    ActionType.PICK_ITEM,
    "DROP_ITEM":    ActionType.DROP_ITEM,
    "RECHARGE":     ActionType.RECHARGE,
}


def _parse_action(text: str) -> ActionType:
    """Extract action from LLM output. Falls back to MOVE_FORWARD on parse error."""
    cleaned = text.strip().upper().replace(" ", "_")
    for key, val in _ACTION_MAP.items():
        if key in cleaned:
            return val
    return ActionType.MOVE_FORWARD


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client: OpenAI,
    task_name: str,
    seed: int,
    verbose: bool = False,
) -> float:
    """Run one episode and return the grade score [0, 1]."""
    env = WarehouseEnv(task_name=task_name, seed=seed)
    obs = env.reset()

    conversation: List[dict] = []
    # [FIX #4] Increased from 4 to 8 — each "turn" is now 3 messages
    # (observation, action, feedback), so this keeps ~24 messages.
    MAX_HISTORY_TURNS = 8

    # [FIX #8] Stuck detection — track recent actions
    recent_actions: List[str] = []
    STUCK_WINDOW = 6

    done = False
    while not done:
        obs_dict = obs.model_dump()
        user_content = _obs_to_prompt(obs_dict)

        conversation.append({"role": "user", "content": user_content})

        # Trim history to last MAX_HISTORY_TURNS complete exchanges.
        # Each exchange = 3 messages: observation + action + feedback
        max_messages = MAX_HISTORY_TURNS * 3
        if len(conversation) > max_messages:
            conversation = conversation[-max_messages:]

        # [FIX #8] Stuck detection: if repeating the same 2 actions, nudge the LLM
        if len(recent_actions) >= STUCK_WINDOW:
            last_n = recent_actions[-STUCK_WINDOW:]
            if len(set(last_n)) <= 2:
                stuck_hint = (
                    "\n[SYSTEM HINT: You have been repeating "
                    f"{', '.join(last_n[-4:])} in a loop. "
                    "Try a COMPLETELY different action to break free. "
                    "If blocked, turn to face a clear path then move.]"
                )
                conversation[-1]["content"] += stuck_hint

        # [FIX #8] Non-zero temperature for exploration
        temperature = 0.2

        # Retry with backoff for rate-limit (429) errors
        raw_action = "MOVE_FORWARD"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                    + conversation,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                )
                raw_action = response.choices[0].message.content or "MOVE_FORWARD"
                break  # success
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str and attempt < max_retries - 1:
                    wait = min(15 * (2 ** attempt), 60)
                    print(
                        f"  [Rate limited] Waiting {wait}s before retry "
                        f"({attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait)
                else:
                    print(f"  [API error] {exc} – defaulting to MOVE_FORWARD")
                    break

        action_type = _parse_action(raw_action)
        action = Action(action_type=action_type)

        conversation.append({"role": "assistant", "content": action_type.name})

        obs, reward, done, info = env.step(action)

        # ===================================================================
        # [FIX #1] Inject feedback into the conversation so the LLM knows
        # whether its action succeeded, what reward it got, and current
        # progress.  THIS IS THE MOST CRITICAL FIX.
        # ===================================================================
        feedback = _build_feedback(reward, info, action_type)
        conversation.append({"role": "user", "content": f"[FEEDBACK] {feedback}"})

        # Track recent actions for stuck detection
        recent_actions.append(action_type.name)
        if len(recent_actions) > STUCK_WINDOW * 2:
            recent_actions = recent_actions[-(STUCK_WINDOW * 2) :]

        if verbose:
            print(
                f"  step={info['step']:03d} "
                f"action={action_type.name:<15} "
                f"reward={reward.total:+.3f} "
                f"delivered={info['delivered_items']}/{info['total_items']} "
                f"battery={info['battery_level']:.1f}%"
            )

    score = env.grade()
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("  Either export it:  export GROQ_API_KEY='gsk_...'")
        print("  Or add it to your .env file.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
    )
    rng = random.Random(BASE_SEED)

    print("=" * 60)
    print("  Autonomous Warehouse Robot – Baseline Evaluation")
    print(f"  Model : {OPENAI_MODEL}")
    print(f"  Seed  : {BASE_SEED}")
    print(f"  Episodes per task: {NUM_EPISODES}")
    print("=" * 60)

    task_averages: Dict[str, float] = {}

    for task_name in TASK_NAMES:
        print(f"\n{'─'*60}")
        print(f"  Task: {task_name.upper()}")
        print(f"{'─'*60}")

        episode_scores: List[float] = []
        for ep in range(NUM_EPISODES):
            ep_seed = rng.randint(0, 2**31 - 1)
            print(f"\n  Episode {ep + 1}/{NUM_EPISODES}  (seed={ep_seed})")
            score = run_episode(
                client=client,
                task_name=task_name,
                seed=ep_seed,
                verbose=True,
            )
            episode_scores.append(score)
            print(f"  → Episode score: {score:.4f}")

        avg = sum(episode_scores) / len(episode_scores)
        task_averages[task_name] = avg
        print(f"\n  ✓ Average score [{task_name}]: {avg:.4f}")

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for task_name, avg in task_averages.items():
        print(f"  {task_name:<8} : {avg:.4f}")
    overall = sum(task_averages.values()) / len(task_averages)
    print(f"  {'overall':<8} : {overall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()