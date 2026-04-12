"""
Reward function for the Autonomous Warehouse Robot environment.
Implements a dense, decomposed reward that provides signal throughout the trajectory.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
from env.models import Reward


# ---------------------------------------------------------------------------
# Reward constants (tune per task if desired)
# ---------------------------------------------------------------------------

STEP_PENALTY          = -0.01   # Every step
APPROACH_SCALE        = 0.05    # Per unit of distance closed toward target
PICKUP_REWARD         = 2.0     # Successfully picking up an item
DELIVERY_REWARD       = 5.0     # Successfully delivering an item
COLLISION_PENALTY     = -1.0    # Hitting a wall or obstacle
INVALID_ACT_PENALTY   = -0.2    # Invalid action (e.g., pick when nothing there)
BATTERY_LOW_PENALTY   = -0.05   # Per step when battery < 20 %
EFFICIENCY_BONUS      = 2.0     # Completing all deliveries under par steps


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def compute_reward(
    *,
    prev_robot_pos: Tuple[float, float],
    curr_robot_pos: Tuple[float, float],
    target_pos: Optional[Tuple[float, float]],
    collision: bool,
    invalid_action: bool,
    picked_up: bool,
    delivered: bool,
    battery_level: float,
    all_delivered: bool,
    steps_elapsed: int,
    par_steps: int,
) -> Reward:
    """
    Compute the full decomposed reward for a single timestep.

    Parameters
    ----------
    prev_robot_pos   : robot position before the action
    curr_robot_pos   : robot position after the action
    target_pos       : current navigation target (None if no active target)
    collision        : True if the action caused a collision
    invalid_action   : True if the action was logically invalid
    picked_up        : True if an item was successfully picked up this step
    delivered        : True if an item was successfully delivered this step
    battery_level    : current battery percentage (0-100)
    all_delivered    : True if all items have been delivered this episode
    steps_elapsed    : steps taken so far (used for efficiency bonus)
    par_steps        : benchmark steps for efficiency bonus
    """
    step_penalty        = STEP_PENALTY
    approach_reward     = 0.0
    pickup_reward       = 0.0
    delivery_reward     = 0.0
    collision_penalty   = 0.0
    invalid_act_penalty = 0.0
    battery_penalty     = 0.0
    efficiency_bonus    = 0.0

    # --- Approach shaping (only when a target is defined and no collision)
    if target_pos is not None and not collision:
        prev_dist = euclidean(prev_robot_pos, target_pos)
        curr_dist = euclidean(curr_robot_pos, target_pos)
        delta = prev_dist - curr_dist          # positive → moved closer
        approach_reward = APPROACH_SCALE * delta

    # --- Collision
    if collision:
        collision_penalty = COLLISION_PENALTY

    # --- Invalid action
    if invalid_action:
        invalid_act_penalty = INVALID_ACT_PENALTY

    # --- Pickup
    if picked_up:
        pickup_reward = PICKUP_REWARD

    # --- Delivery
    if delivered:
        delivery_reward = DELIVERY_REWARD

    # --- Battery low penalty
    if battery_level < 20.0:
        battery_penalty = BATTERY_LOW_PENALTY

    # --- Efficiency bonus (only once, when all delivered under par)
    if all_delivered and steps_elapsed <= par_steps:
        efficiency_bonus = EFFICIENCY_BONUS

    total = (
        step_penalty
        + approach_reward
        + pickup_reward
        + delivery_reward
        + collision_penalty
        + invalid_act_penalty
        + battery_penalty
        + efficiency_bonus
    )

    return Reward(
        total=round(total, 6),
        step_penalty=step_penalty,
        approach_reward=round(approach_reward, 6),
        pickup_reward=pickup_reward,
        delivery_reward=delivery_reward,
        collision_penalty=collision_penalty,
        invalid_action_penalty=invalid_act_penalty,
        battery_penalty=battery_penalty,
        efficiency_bonus=efficiency_bonus,
    )