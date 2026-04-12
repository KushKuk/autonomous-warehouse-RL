"""
Pydantic models for the Autonomous Warehouse Robot environment.
Defines typed Observation, Action, and Reward structures per OpenEnv spec.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from enum import IntEnum
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    MOVE_FORWARD = 0
    TURN_LEFT    = 1
    TURN_RIGHT   = 2
    PICK_ITEM    = 3
    DROP_ITEM    = 4
    RECHARGE     = 5


class Action(BaseModel):
    """Typed action that the agent sends to the environment."""
    action_type: ActionType = Field(..., description="Discrete action identifier")

    class Config:
        use_enum_values = False


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class RobotState(BaseModel):
    """Core robot kinematics and resource state."""
    position: Tuple[float, float] = Field(..., description="(x, y) in grid coords")
    velocity: Tuple[float, float] = Field(..., description="(vx, vy) current velocity vector")
    heading: int = Field(..., description="Direction: 0=North,1=East,2=South,3=West")
    battery_level: float = Field(..., ge=0.0, le=100.0, description="Battery % remaining")
    carrying_item: bool = Field(..., description="Whether robot holds an item")
    carried_item_id: Optional[int] = Field(None, description="ID of carried item, if any")


class ObstacleInfo(BaseModel):
    """Info about a single nearby obstacle."""
    position: Tuple[int, int]
    is_dynamic: bool


class ItemInfo(BaseModel):
    """Info about a pickable item in the warehouse."""
    item_id: int
    position: Tuple[int, int]
    is_picked: bool
    target_station: int = Field(..., description="Station ID where this item must be delivered")


class StationInfo(BaseModel):
    """Delivery station info."""
    station_id: int
    position: Tuple[int, int]


class Observation(BaseModel):
    """Full structured observation returned by reset() and step()."""
    robot: RobotState
    grid_map: List[List[int]] = Field(
        ...,
        description=(
            "Encoded grid. 0=free, 1=wall/shelf, 2=delivery station, "
            "3=charging station, 4=dynamic obstacle"
        ),
    )
    nearby_obstacles: List[ObstacleInfo] = Field(
        ..., description="Obstacles within sensor range of robot"
    )
    item_locations: List[ItemInfo] = Field(..., description="All items and their states")
    stations: List[StationInfo] = Field(..., description="All delivery stations")
    current_target: Optional[Tuple[int, int]] = Field(
        None, description="Current navigation waypoint"
    )
    steps_remaining: int = Field(..., description="Steps left in episode")
    episode_time: int = Field(..., description="Steps elapsed since reset")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Decomposed reward signal for transparency and debugging."""
    total: float = Field(..., description="Scalar reward passed to the agent")

    # Components
    step_penalty: float = Field(0.0, description="Small penalty every step")
    approach_reward: float = Field(0.0, description="Shaping: moving closer to target")
    pickup_reward: float = Field(0.0, description="Bonus for successful pickup")
    delivery_reward: float = Field(0.0, description="Bonus for successful delivery")
    collision_penalty: float = Field(0.0, description="Penalty for hitting obstacles/walls")
    invalid_action_penalty: float = Field(0.0, description="Penalty for impossible actions")
    battery_penalty: float = Field(0.0, description="Penalty for running low on battery")
    efficiency_bonus: float = Field(0.0, description="Bonus for completing under par steps")


# ---------------------------------------------------------------------------
# Full internal state (returned by state())
# ---------------------------------------------------------------------------

class InternalState(BaseModel):
    """Complete environment state snapshot (for debugging / recording)."""
    observation: Observation
    total_items: int
    delivered_items: int
    collision_count: int
    invalid_action_count: int
    cumulative_reward: float
    done: bool
    task_name: str
    seed: int