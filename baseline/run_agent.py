"""
Baseline inference script for the Autonomous Warehouse Robot environment.

Uses the OpenAI API to run an LLM-guided agent on all three task difficulties
(easy, medium, hard) and prints per-task scores.

Usage
-----
    OPENAI_API_KEY=<your-key> python baseline/run_agent.py

Environment variables
---------------------
    OPENAI_API_KEY   : required - your OpenAI API key
    OPENAI_MODEL     : optional - model name (default: gpt-4o-mini)
    NUM_EPISODES     : optional - episodes per task (default: 3)
    SEED             : optional - base random seed (default: 42)
"""

from __future__ import annotations

import json
import os
import sys
import random
import textwrap
from pathlib import Path
from typing import Dict, List

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

from env.environment import WarehouseEnv
from env.models import Action, ActionType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NUM_EPISODES  = int(os.getenv("NUM_EPISODES", "3"))
BASE_SEED     = int(os.getenv("SEED", "42"))
TASK_NAMES    = ["easy", "medium", "hard"]
MAX_TOKENS    = 256


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous warehouse robot controller.
    You receive structured JSON observations and must output a single action.

    Available actions (output ONLY the action name, nothing else):
      MOVE_FORWARD   – move one cell in the current heading direction
      TURN_LEFT      – rotate 90° counter-clockwise
      TURN_RIGHT     – rotate 90° clockwise
      PICK_ITEM      – pick up an item at the current cell
      DROP_ITEM      – drop the carried item at a delivery station
      RECHARGE       – recharge battery if on a charging station

    Strategy:
    1. Navigate toward the nearest unpicked item (use current_target).
    2. Pick it up (PICK_ITEM) when at the item's position.
    3. Navigate toward the designated delivery station.
    4. Drop it (DROP_ITEM) when at the station.
    5. Repeat until all items are delivered.
    6. Avoid walls (grid cell = 1) and dynamic obstacles (grid cell = 4).
    7. Recharge when battery is low and you are on a charging station (grid cell = 3).

    Output ONLY one of the action names listed above. No explanation.
""")


def _obs_to_prompt(obs_dict: dict) -> str:
    """Convert observation dict to a concise LLM prompt."""
    robot   = obs_dict["robot"]
    target  = obs_dict["current_target"]
    items   = [
        {"id": it["item_id"], "pos": it["position"], "picked": it["is_picked"]}
        for it in obs_dict["item_locations"]
    ]
    nearby  = [
        {"pos": ob["position"], "dynamic": ob["is_dynamic"]}
        for ob in obs_dict["nearby_obstacles"]
    ]
    stations = [
        {"id": st["station_id"], "pos": st["position"]}
        for st in obs_dict["stations"]
    ]

    summary = {
        "robot_position":   robot["position"],
        "robot_heading":    robot["heading"],   # 0=N,1=E,2=S,3=W
        "battery":          robot["battery_level"],
        "carrying_item":    robot["carrying_item"],
        "carried_item_id":  robot["carried_item_id"],
        "current_target":   target,
        "items":            items,
        "nearby_obstacles": nearby,
        "delivery_stations": stations,
        "steps_remaining":  obs_dict["steps_remaining"],
    }
    return json.dumps(summary, indent=2)


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

    conversation: List[dict] = []  # maintain rolling history (last N turns)
    MAX_HISTORY_TURNS = 4          # keep context window manageable

    done = False
    while not done:
        obs_dict = obs.model_dump()
        user_content = _obs_to_prompt(obs_dict)

        conversation.append({"role": "user", "content": user_content})

        # Trim history to last MAX_HISTORY_TURNS complete turns
        if len(conversation) > MAX_HISTORY_TURNS * 2:
            conversation = conversation[-(MAX_HISTORY_TURNS * 2):]

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                max_tokens=MAX_TOKENS,
                temperature=0.0,          # deterministic
                seed=seed,
            )
            raw_action = response.choices[0].message.content or "MOVE_FORWARD"
        except Exception as exc:
            print(f"  [OpenAI error] {exc} – defaulting to MOVE_FORWARD")
            raw_action = "MOVE_FORWARD"

        action_type = _parse_action(raw_action)
        action = Action(action_type=action_type)

        conversation.append({"role": "assistant", "content": action_type.name})

        obs, reward, done, info = env.step(action)

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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
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