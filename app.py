"""
Autonomous Warehouse Robot – Hugging Face Spaces Demo
=====================================================
Launch with:  python app.py
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── make sure project root is on sys.path ──────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from env.environment import WarehouseEnv
from env.models import Action, ActionType

# ── try loading .env (optional) ────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_BASE_URL  = "https://api.groq.com/openai/v1"
DEFAULT_MODEL     = "llama3-70b-8192"
TASK_NAMES        = ["easy", "medium", "hard"]
MAX_TOKENS        = 256

# ── colour palette for grid rendering ─────────────────────────────────────
_CELL_COLORS = {
    0: "#1a1a2e",   # free floor – dark navy
    1: "#4a4e69",   # shelf / wall – grey-purple
    2: "#f72585",   # delivery station – vivid pink
    3: "#4cc9f0",   # charging station – sky blue
    4: "#f77f00",   # dynamic obstacle – amber
}
_ROBOT_COLOR  = "#7bed9f"   # robot – mint green


# ═══════════════════════════════════════════════════════════════════════════
# Grid rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_grid(
    grid: List[List[int]],
    robot_pos: Tuple[int, int],
    items: List[dict],
    stations: List[dict],
    title: str = "",
) -> "PIL.Image.Image":
    """Render the warehouse grid as a PIL image using matplotlib."""
    from PIL import Image as PILImage

    rows = len(grid)
    cols = len(grid[0]) if rows else 1

    cell_px = max(32, min(64, 512 // max(rows, cols)))
    fig_w = cols * cell_px / 96
    fig_h = rows * cell_px / 96

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=96)
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            color = _CELL_COLORS.get(grid[r][c], "#1a1a2e")
            rect = plt.Rectangle(
                (c, rows - 1 - r), 1, 1,
                linewidth=0.3, edgecolor="#2a2a40", facecolor=color,
            )
            ax.add_patch(rect)

    # Draw items (small yellow circles)
    for it in items:
        if not it.get("is_picked", False):
            r, c = it["position"]
            ax.add_patch(
                plt.Circle(
                    (c + 0.5, rows - 1 - r + 0.5), 0.25,
                    color="#ffe66d", zorder=3,
                )
            )

    # Draw robot
    rr, rc = robot_pos
    ax.add_patch(
        plt.Circle(
            (rc + 0.5, rows - 1 - rr + 0.5), 0.35,
            color=_ROBOT_COLOR, zorder=4,
        )
    )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_CELL_COLORS[1], label="Shelf"),
        mpatches.Patch(facecolor=_CELL_COLORS[2], label="Station"),
        mpatches.Patch(facecolor=_CELL_COLORS[3], label="Charger"),
        mpatches.Patch(facecolor=_CELL_COLORS[4], label="Dyn. Obstacle"),
        mpatches.Patch(facecolor=_ROBOT_COLOR,     label="Robot"),
        mpatches.Patch(facecolor="#ffe66d",         label="Item"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=6,
        facecolor="#1a1a2e",
        labelcolor="white",
        framealpha=0.8,
        ncol=3,
        bbox_to_anchor=(0, -0.01),
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


# ═══════════════════════════════════════════════════════════════════════════
# LLM helpers (adapted from baseline/run_agent.py)
# ═══════════════════════════════════════════════════════════════════════════

_HEADING_NAMES = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
_HEADING_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1,  0),
    1:  (0,  1),
    2:  (1,  0),
    3:  (0, -1),
}

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
    1. Read the navigation hints — they tell you which direction the target is.
    2. Check "cell_ahead" before using MOVE_FORWARD. Turn first if WALL or OBSTACLE.
    3. Use [FEEDBACK] messages to verify your actions succeeded.
    4. After picking an item, navigate to its target_station to drop it.
    5. If battery < 20%, navigate to a charging station and RECHARGE.

    Output ONLY one action name. No explanation, no punctuation.
""")

_ACTION_MAP: Dict[str, ActionType] = {
    "MOVE_FORWARD": ActionType.MOVE_FORWARD,
    "TURN_LEFT":    ActionType.TURN_LEFT,
    "TURN_RIGHT":   ActionType.TURN_RIGHT,
    "PICK_ITEM":    ActionType.PICK_ITEM,
    "DROP_ITEM":    ActionType.DROP_ITEM,
    "RECHARGE":     ActionType.RECHARGE,
}


def _parse_action(text: str) -> ActionType:
    cleaned = text.strip().upper().replace(" ", "_")
    for key, val in _ACTION_MAP.items():
        if key in cleaned:
            return val
    return ActionType.MOVE_FORWARD


def _compute_direction(robot_pos, target_pos) -> str:
    dr = target_pos[0] - robot_pos[0]
    dc = target_pos[1] - robot_pos[1]
    parts = []
    if dr < -0.5:  parts.append("NORTH")
    elif dr > 0.5: parts.append("SOUTH")
    if dc < -0.5:  parts.append("WEST")
    elif dc > 0.5: parts.append("EAST")
    return "-".join(parts) if parts else "AT_TARGET"


def _suggest_action(robot_pos, robot_heading, target_pos, grid) -> str:
    rr, rc = int(robot_pos[0]), int(robot_pos[1])
    tr, tc = int(target_pos[0]), int(target_pos[1])
    dr, dc = tr - rr, tc - rc
    if dr == 0 and dc == 0:
        return "AT_TARGET"
    ideal = (2 if dr > 0 else 0) if abs(dr) >= abs(dc) else (1 if dc > 0 else 3)
    if robot_heading == ideal:
        hdr, hdc = _HEADING_DELTA[robot_heading]
        nr, nc = rr + hdr, rc + hdc
        rows, cols = len(grid), len(grid[0]) if grid else 0
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] not in (1, 4):
            return "MOVE_FORWARD"
        return "TURN_RIGHT"
    diff = (ideal - robot_heading) % 4
    return "TURN_RIGHT" if diff == 1 else ("TURN_LEFT" if diff == 3 else "TURN_RIGHT")


def _extract_local_grid(grid, robot_pos, radius=3):
    rr, rc = int(robot_pos[0]), int(robot_pos[1])
    rows, cols = len(grid), len(grid[0]) if grid else 0
    local = []
    for r in range(rr - radius, rr + radius + 1):
        row = []
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
    robot  = obs_dict["robot"]
    target = obs_dict["current_target"]
    grid   = obs_dict["grid_map"]

    robot_pos = (robot["position"][0], robot["position"][1])
    heading   = robot["heading"]

    items = [
        {"id": it["item_id"], "pos": it["position"],
         "picked": it["is_picked"], "target_station": it["target_station"]}
        for it in obs_dict["item_locations"]
    ]
    nearby   = [{"pos": ob["position"], "dynamic": ob["is_dynamic"]}
                for ob in obs_dict["nearby_obstacles"]]
    stations = [{"id": st["station_id"], "pos": st["position"]}
                for st in obs_dict["stations"]]

    local_grid = _extract_local_grid(grid, robot_pos, radius=3)

    nav_hints: Dict = {}
    if target is not None:
        target_t = (target[0], target[1])
        nav_hints["direction_to_target"] = _compute_direction(robot_pos, target_t)
        nav_hints["suggested_action"]    = _suggest_action(robot_pos, heading, target_t, grid)
        nav_hints["distance"]            = round(
            math.sqrt((target[0]-robot_pos[0])**2 + (target[1]-robot_pos[1])**2), 1
        )
    else:
        nav_hints = {"direction_to_target": "ALL_DELIVERED", "suggested_action": "NONE", "distance": 0}

    hdr, hdc = _HEADING_DELTA[heading]
    ahead_r, ahead_c = int(robot_pos[0]) + hdr, int(robot_pos[1]) + hdc
    _cell_names = {0: "free", 1: "WALL", 2: "station", 3: "charger", 4: "OBSTACLE"}
    rows, cols = len(grid), len(grid[0]) if grid else 0
    cell_ahead = (_cell_names.get(grid[ahead_r][ahead_c], "unknown")
                  if 0 <= ahead_r < rows and 0 <= ahead_c < cols else "OUT_OF_BOUNDS")

    summary = {
        "robot_position":    list(robot_pos),
        "robot_heading":     f"{heading} ({_HEADING_NAMES[heading]})",
        "cell_ahead":        cell_ahead,
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


def _build_feedback(reward, info: dict, action_type: ActionType) -> str:
    parts = [f"Action {action_type.name}", f"reward={reward.total:+.3f}"]
    details = []
    if reward.collision_penalty != 0:       details.append(f"COLLISION({reward.collision_penalty:+.1f})")
    if reward.invalid_action_penalty != 0:  details.append(f"INVALID_ACTION({reward.invalid_action_penalty:+.1f})")
    if reward.pickup_reward != 0:           details.append(f"PICKUP_SUCCESS(+{reward.pickup_reward:.1f})")
    if reward.delivery_reward != 0:         details.append(f"DELIVERY_SUCCESS(+{reward.delivery_reward:.1f})")
    if reward.approach_reward > 0.001:      details.append("moved_closer")
    elif reward.approach_reward < -0.001:   details.append("moved_AWAY")
    if reward.battery_penalty != 0:         details.append("LOW_BATTERY_WARNING")
    if reward.efficiency_bonus != 0:        details.append(f"EFFICIENCY_BONUS(+{reward.efficiency_bonus:.1f})")
    if details:
        parts.append(" | ".join(details))
    parts.append(f"delivered={info['delivered_items']}/{info['total_items']} battery={info['battery_level']:.1f}%")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Episode runner (generator — yields (log_line, grid_img) each step)
# ═══════════════════════════════════════════════════════════════════════════

def run_episode_streaming(
    client,
    task_name: str,
    seed: int,
    model: str,
) -> Generator[Tuple[str, Optional["PIL.Image.Image"]], None, float]:
    """
    Yields (log_line, grid_image) for every step.
    The final yield carries the episode score as a third element.
    """
    env = WarehouseEnv(task_name=task_name, seed=seed)
    obs = env.reset()

    conversation: List[dict] = []
    MAX_HISTORY_TURNS = 8
    recent_actions: List[str] = []
    STUCK_WINDOW = 6

    done = False
    while not done:
        obs_dict = obs.model_dump()
        user_content = _obs_to_prompt(obs_dict)
        conversation.append({"role": "user", "content": user_content})

        max_messages = MAX_HISTORY_TURNS * 3
        if len(conversation) > max_messages:
            conversation = conversation[-max_messages:]

        if len(recent_actions) >= STUCK_WINDOW:
            last_n = recent_actions[-STUCK_WINDOW:]
            if len(set(last_n)) <= 2:
                stuck_hint = (
                    "\n[SYSTEM HINT: You have been repeating "
                    f"{', '.join(last_n[-4:])} in a loop. "
                    "Try a COMPLETELY different action.]"
                )
                conversation[-1]["content"] += stuck_hint

        raw_action = "MOVE_FORWARD"
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                    max_tokens=MAX_TOKENS,
                    temperature=0.2,
                )
                raw_action = response.choices[0].message.content or "MOVE_FORWARD"
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 2:
                    wait = min(15 * (2 ** attempt), 60)
                    yield f"  [Rate limited] Waiting {wait}s…", None
                    time.sleep(wait)
                else:
                    yield f"  [API error] {exc} – using MOVE_FORWARD", None
                    break

        action_type = _parse_action(raw_action)
        action = Action(action_type=action_type)
        conversation.append({"role": "assistant", "content": action_type.name})

        obs, reward, done, info = env.step(action)

        feedback = _build_feedback(reward, info, action_type)
        conversation.append({"role": "user", "content": f"[FEEDBACK] {feedback}"})

        recent_actions.append(action_type.name)
        if len(recent_actions) > STUCK_WINDOW * 2:
            recent_actions = recent_actions[-(STUCK_WINDOW * 2):]

        log_line = (
            f"  step={info['step']:03d}  {action_type.name:<15} "
            f"reward={reward.total:+.3f}  "
            f"delivered={info['delivered_items']}/{info['total_items']}  "
            f"battery={info['battery_level']:.1f}%"
        )

        # Render grid every 5 steps to avoid overwhelming the UI
        grid_img = None
        if info["step"] % 5 == 0 or done:
            obs_dict2 = obs.model_dump()
            robot_pos = (int(obs_dict2["robot"]["position"][0]),
                         int(obs_dict2["robot"]["position"][1]))
            grid_img = render_grid(
                obs_dict2["grid_map"],
                robot_pos,
                obs_dict2["item_locations"],
                obs_dict2["stations"],
                title=f"Task: {task_name.upper()} | Step {info['step']}",
            )

        yield log_line, grid_img

    score = env.grade()
    yield f"\n  ✓ Episode score: {score:.4f}", None, score


# ═══════════════════════════════════════════════════════════════════════════
# Gradio UI entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def run_demo(
    api_key: str,
    base_url: str,
    model: str,
    tasks_selected: List[str],
    num_episodes: int,
    seed: int,
) -> Generator:
    """Main generator fed into Gradio's streaming interface."""
    from openai import OpenAI

    if not api_key.strip():
        yield "❌ Please enter your API key.", None, []
        return

    if not tasks_selected:
        yield "❌ Please select at least one task.", None, []
        return

    client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip() or DEFAULT_BASE_URL)
    rng    = random.Random(seed)

    full_log     = ""
    latest_img   = None
    results_data: List[List] = []

    header = (
        "=" * 60 + "\n"
        "  Autonomous Warehouse Robot – Baseline Evaluation\n"
        f"  Model   : {model}\n"
        f"  Seed    : {seed}\n"
        f"  Episodes: {num_episodes} per task\n"
        "=" * 60 + "\n"
    )
    full_log += header
    yield full_log, None, results_data

    for task_name in tasks_selected:
        full_log += f"\n{'─'*60}\n  Task: {task_name.upper()}\n{'─'*60}\n"
        yield full_log, latest_img, results_data

        episode_scores: List[float] = []

        for ep in range(int(num_episodes)):
            ep_seed = rng.randint(0, 2**31 - 1)
            full_log += f"\n  Episode {ep+1}/{num_episodes}  (seed={ep_seed})\n"
            yield full_log, latest_img, results_data

            ep_log = ""
            ep_score = 0.0

            gen = run_episode_streaming(client, task_name, ep_seed, model)
            for packet in gen:
                if len(packet) == 3:
                    line, img, ep_score = packet
                else:
                    line, img = packet

                ep_log += line + "\n"
                full_log_display = full_log + ep_log
                if img is not None:
                    latest_img = img
                yield full_log_display, latest_img, results_data

            full_log += ep_log
            episode_scores.append(ep_score)
            results_data.append([task_name, ep + 1, ep_seed, f"{ep_score:.4f}"])
            yield full_log, latest_img, results_data

        avg = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        full_log += f"\n  ✓ Average [{task_name}]: {avg:.4f}\n"
        results_data.append([task_name, "AVERAGE", "—", f"{avg:.4f}"])
        yield full_log, latest_img, results_data

    full_log += "\n" + "=" * 60 + "\n  Run complete!\n" + "=" * 60 + "\n"
    yield full_log, latest_img, results_data


# ═══════════════════════════════════════════════════════════════════════════
# Build the Gradio interface
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
body { background: #0d0d1a; }
#title-row { text-align: center; }
#title-row h1 { font-size: 2rem; background: linear-gradient(90deg, #4cc9f0, #f72585); 
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
#title-row p  { color: #8d99ae; font-size: 0.95rem; }
.gradio-container { max-width: 1200px; margin: auto; }
#log-box textarea { font-family: 'Courier New', monospace; font-size: 0.8rem;
                    background: #0d0d1a; color: #7bed9f; border: 1px solid #2a2a40; }
#run-btn { background: linear-gradient(135deg, #4361ee, #7209b7) !important; border: none !important; }
#run-btn:hover { opacity: 0.85; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="🤖 Autonomous Warehouse Robot") as demo:

    with gr.Row(elem_id="title-row"):
        with gr.Column():
            gr.HTML("""
                <h1>🤖 Autonomous Warehouse Robot</h1>
                <p>An LLM-guided agent navigating a 2-D warehouse grid.
                   Pick items, avoid obstacles, and deliver to stations — all controlled by a language model.</p>
            """)

    with gr.Row():
        # ── Left panel: controls ──────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### ⚙️ Configuration")

            api_key_input = gr.Textbox(
                label="API Key (Groq / OpenAI-compatible) 🔑",
                placeholder="gsk_... or sk-...",
                type="password",
                info="Your key is never stored or logged.",
            )
            base_url_input = gr.Textbox(
                label="Base URL",
                value=DEFAULT_BASE_URL,
                info="Change to use a different OpenAI-compatible endpoint.",
            )
            model_input = gr.Textbox(
                label="Model",
                value=DEFAULT_MODEL,
                info="e.g. llama3-70b-8192, gpt-4o-mini, gemini-2.0-flash",
            )

            gr.Markdown("### 🎮 Task Settings")

            tasks_input = gr.CheckboxGroup(
                choices=TASK_NAMES,
                value=["easy"],
                label="Tasks to run",
            )
            num_episodes_input = gr.Slider(
                minimum=1, maximum=5, value=1, step=1,
                label="Episodes per task",
            )
            seed_input = gr.Number(value=42, label="Base seed", precision=0)

            run_btn = gr.Button("▶ Run Agent", variant="primary", elem_id="run-btn")

            gr.Markdown("""
---
**Grid legend:**
🟩 Robot &nbsp; 🟡 Item &nbsp; 🩷 Station  
🔵 Charger &nbsp; 🟠 Dyn. Obstacle &nbsp; ⬛ Shelf  
""")

        # ── Right panel: outputs ──────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 🗺️ Warehouse Grid")
            grid_output = gr.Image(
                label="Live grid view (updates every 5 steps)",
                type="pil",
                height=380,
                show_label=False,
            )

            gr.Markdown("### 📋 Live Log")
            log_output = gr.Textbox(
                label="",
                lines=18,
                max_lines=18,
                interactive=False,
                elem_id="log-box",
                show_copy_button=True,
            )

            gr.Markdown("### 📊 Results")
            results_output = gr.Dataframe(
                headers=["Task", "Episode", "Seed", "Score"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

    run_btn.click(
        fn=run_demo,
        inputs=[
            api_key_input,
            base_url_input,
            model_input,
            tasks_input,
            num_episodes_input,
            seed_input,
        ],
        outputs=[log_output, grid_output, results_output],
    )

if __name__ == "__main__":
    demo.launch()
