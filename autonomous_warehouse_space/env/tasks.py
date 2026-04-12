"""
Task definitions and graders for the Autonomous Warehouse Robot environment.

Three difficulty levels:
  - easy   : single item, no dynamic obstacles, small grid
  - medium : multiple items, static obstacles, moderate grid
  - hard   : multiple items, dynamic obstacles, battery constraints, time limit
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Task configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    name: str
    grid_rows: int
    grid_cols: int
    num_items: int
    num_static_obstacles: int
    num_dynamic_obstacles: int
    num_stations: int
    num_charging_stations: int
    max_steps: int
    battery_drain_per_step: float    # % per step
    battery_start: float             # starting battery %
    battery_enabled: bool            # whether battery depletion ends episode
    dynamic_obs_speed: int           # how often dynamic obs move (every N steps)
    par_steps: int                   # steps to beat for efficiency bonus
    sensor_range: int                # how many cells around robot we expose
    seed_offset: int                 # added to episode seed for reproducibility


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        grid_rows=8,
        grid_cols=8,
        num_items=1,
        num_static_obstacles=3,
        num_dynamic_obstacles=0,
        num_stations=1,
        num_charging_stations=1,
        max_steps=100,
        battery_drain_per_step=0.3,
        battery_start=100.0,
        battery_enabled=False,
        dynamic_obs_speed=0,
        par_steps=30,
        sensor_range=3,
        seed_offset=0,
    ),
    "medium": TaskConfig(
        name="medium",
        grid_rows=12,
        grid_cols=12,
        num_items=3,
        num_static_obstacles=10,
        num_dynamic_obstacles=0,
        num_stations=2,
        num_charging_stations=1,
        max_steps=300,
        battery_drain_per_step=0.2,
        battery_start=100.0,
        battery_enabled=False,
        dynamic_obs_speed=0,
        par_steps=150,
        sensor_range=4,
        seed_offset=1000,
    ),
    "hard": TaskConfig(
        name="hard",
        grid_rows=16,
        grid_cols=16,
        num_items=5,
        num_static_obstacles=20,
        num_dynamic_obstacles=4,
        num_stations=3,
        num_charging_stations=2,
        max_steps=500,
        battery_drain_per_step=0.4,
        battery_start=100.0,
        battery_enabled=True,
        dynamic_obs_speed=5,
        par_steps=300,
        sensor_range=5,
        seed_offset=9000,
    ),
}


# ---------------------------------------------------------------------------
# Grader functions
# ---------------------------------------------------------------------------

def _path_efficiency_score(steps_taken: int, par_steps: int) -> float:
    """
    Returns a score in [0, 1] based on how close steps_taken is to par_steps.
    At or below par → 1.0. Degrades linearly up to 2× par, then 0.
    """
    if steps_taken <= par_steps:
        return 1.0
    overshoot = steps_taken - par_steps
    return max(0.0, 1.0 - overshoot / par_steps)


def grade_episode(
    *,
    task_name: str,
    total_items: int,
    delivered_items: int,
    collision_count: int,
    invalid_action_count: int,
    steps_taken: int,
    battery_depleted: bool,
) -> float:
    """
    Deterministic grader returning a score in [0.0, 1.0].

    Scoring breakdown:
      - Delivery ratio        : 50 % weight
      - Path efficiency       : 25 % weight
      - Collision penalty     : 15 % weight  (max penalty caps at full weight)
      - Invalid action penalty: 10 % weight
    """
    cfg = TASKS[task_name]

    # 1. Delivery ratio  (0 → 1)
    delivery_ratio = delivered_items / max(total_items, 1)

    # 2. Path efficiency (0 → 1) — only counts if at least one item delivered
    if delivered_items > 0:
        efficiency = _path_efficiency_score(steps_taken, cfg.par_steps)
    else:
        efficiency = 0.0

    # 3. Collision score  — penalise up to full weight
    #    10 collisions → score component = 0
    collision_score = max(0.0, 1.0 - collision_count / 10.0)

    # 4. Invalid action score
    #    20 invalid actions → score component = 0
    invalid_score = max(0.0, 1.0 - invalid_action_count / 20.0)

    # 5. Battery depletion hard penalty
    battery_factor = 0.8 if battery_depleted else 1.0

    raw = (
        0.50 * delivery_ratio
        + 0.25 * efficiency
        + 0.15 * collision_score
        + 0.10 * invalid_score
    )

    return round(min(1.0, raw * battery_factor), 4)