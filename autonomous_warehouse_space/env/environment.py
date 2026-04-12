"""
Autonomous Warehouse Robot - OpenEnv-compliant environment.

The agent navigates a 2-D grid warehouse, picks items from shelves,
avoids static and dynamic obstacles, and delivers items to stations
while managing battery level and path efficiency.

Grid cell encoding
------------------
  0  free floor
  1  shelf / static wall
  2  delivery station
  3  charging station
  4  dynamic obstacle (updated each step)
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from env.models import (
    Action,
    ActionType,
    InternalState,
    ItemInfo,
    Observation,
    ObstacleInfo,
    Reward,
    RobotState,
    StationInfo,
)
from env.rewards import compute_reward
from env.tasks import TASKS, TaskConfig, grade_episode

# Heading → (row_delta, col_delta)
_HEADING_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # North
    1:  (0,  1),   # East
    2:  (1,  0),   # South
    3:  (0, -1),   # West
}

# Dynamic obstacle random walk directions
_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# ---------------------------------------------------------------------------
# Helper: grid layout generation
# ---------------------------------------------------------------------------

def _build_shelf_grid(rows: int, cols: int, rng: random.Random) -> List[List[int]]:
    """
    Generate a realistic warehouse layout:
    - Outer boundary walls (cell = 1)
    - Interior paired shelf rows with clear aisle columns between them
    - Guaranteed horizontal aisle at rows 1 and rows-2 for navigation
    """
    grid = [[0] * cols for _ in range(rows)]

    # Outer boundary
    for r in range(rows):
        grid[r][0] = 1
        grid[r][cols - 1] = 1
    for c in range(cols):
        grid[0][c] = 1
        grid[rows - 1][c] = 1

    # Shelf columns: place pairs of shelf columns every 4 cols
    # Aisles are the columns between them
    for c in range(2, cols - 2, 4):
        for r in range(2, rows - 2):
            # Leave gaps at top/bottom of each shelf column for navigation
            if r > 2 and r < rows - 3 and rng.random() < 0.65:
                grid[r][c] = 1
            # Second shelf column next to first (pair)
            if c + 1 < cols - 1:
                if r > 2 and r < rows - 3 and rng.random() < 0.65:
                    grid[r][c + 1] = 1

    # Guarantee clear horizontal aisle at row 1 (already clear) and a mid-aisle
    mid_row = rows // 2
    for c in range(1, cols - 1):
        grid[mid_row][c] = 0   # clear horizontal aisle

    return grid


class WarehouseEnv:
    """
    OpenEnv-compliant Autonomous Warehouse Robot environment.

    Usage
    -----
    env = WarehouseEnv(task_name="medium", seed=42)
    obs = env.reset()
    while True:
        action = Action(action_type=ActionType.MOVE_FORWARD)
        obs, reward, done, info = env.step(action)
        if done:
            break
    score = env.grade()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, task_name: str = "easy", seed: int = 42) -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from {list(TASKS)}")
        self.task_name: str = task_name
        self.cfg: TaskConfig = TASKS[task_name]
        self.base_seed: int = seed + self.cfg.seed_offset
        self._rng = random.Random(self.base_seed)

        # These are populated by reset()
        self._grid: List[List[int]] = []
        self._robot_pos: Tuple[int, int] = (1, 1)
        self._robot_heading: int = 1          # East
        self._battery: float = self.cfg.battery_start
        self._carrying: bool = False
        self._carried_item_id: Optional[int] = None
        self._items: List[ItemInfo] = []
        self._stations: List[StationInfo] = []
        self._charging_stations: List[Tuple[int, int]] = []
        self._dynamic_obs: List[Tuple[int, int]] = []
        self._step_count: int = 0

        # Episode statistics
        self._delivered_items: int = 0
        self._collision_count: int = 0
        self._invalid_action_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._battery_depleted: bool = False

        # Shaping: previous distance to current target
        self._prev_target_dist: Optional[float] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Observation:
        """Reset the environment and return the initial observation."""
        seed = kwargs.get("seed")
        if seed is not None:
            self.base_seed = seed
        self._rng = random.Random(self.base_seed)

        # Build the static grid layout
        self._grid = _build_shelf_grid(
            self.cfg.grid_rows, self.cfg.grid_cols, self._rng
        )

        # Place delivery stations
        self._stations = self._place_stations()
        for st in self._stations:
            self._grid[st.position[0]][st.position[1]] = 2

        # Place charging stations
        self._charging_stations = self._place_charging_stations()
        for pos in self._charging_stations:
            self._grid[pos[0]][pos[1]] = 3

        # Place static obstacles (random free cells)
        self._place_static_obstacles()

        # Place robot on a free cell with at least one navigable neighbour
        self._robot_pos = self._random_free_cell(require_neighbour=True)
        self._robot_heading = 1  # East
        self._battery = self.cfg.battery_start
        self._carrying = False
        self._carried_item_id = None

        # Place items on shelf-adjacent free cells
        self._items = self._place_items()

        # Place dynamic obstacles on cells with navigable neighbours
        self._dynamic_obs = [
            self._random_free_cell(require_neighbour=True)
            for _ in range(self.cfg.num_dynamic_obstacles)
        ]
        self._update_dynamic_on_grid()

        # Episode counters
        self._step_count = 0
        self._delivered_items = 0
        self._collision_count = 0
        self._invalid_action_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._battery_depleted = False
        self._prev_target_dist = None

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one timestep.

        Parameters
        ----------
        action : Action – typed action from the agent

        Returns
        -------
        (Observation, Reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() first.")

        self._step_count += 1

        prev_pos = self._robot_pos
        collision = False
        invalid_action = False
        picked_up = False
        delivered = False

        act = action.action_type

        # ----------------------------------------------------------------
        # Execute action
        # ----------------------------------------------------------------
        if act == ActionType.MOVE_FORWARD:
            collision = self._try_move_forward()

        elif act == ActionType.TURN_LEFT:
            self._robot_heading = (self._robot_heading - 1) % 4

        elif act == ActionType.TURN_RIGHT:
            self._robot_heading = (self._robot_heading + 1) % 4

        elif act == ActionType.PICK_ITEM:
            result = self._try_pick()
            if result:
                picked_up = True
            else:
                invalid_action = True

        elif act == ActionType.DROP_ITEM:
            result = self._try_drop()
            if result:
                delivered = True
                self._delivered_items += 1
            else:
                invalid_action = True

        elif act == ActionType.RECHARGE:
            result = self._try_recharge()
            if not result:
                invalid_action = True

        # ----------------------------------------------------------------
        # Battery drain
        # ----------------------------------------------------------------
        if act not in (ActionType.RECHARGE,):
            self._battery = max(0.0, self._battery - self.cfg.battery_drain_per_step)

        if self._battery <= 0.0 and self.cfg.battery_enabled:
            self._battery_depleted = True

        # ----------------------------------------------------------------
        # Move dynamic obstacles
        # ----------------------------------------------------------------
        if (
            self.cfg.num_dynamic_obstacles > 0
            and self.cfg.dynamic_obs_speed > 0
            and self._step_count % self.cfg.dynamic_obs_speed == 0
        ):
            self._move_dynamic_obstacles()

        # ----------------------------------------------------------------
        # Collision statistics
        # ----------------------------------------------------------------
        if collision:
            self._collision_count += 1
        if invalid_action:
            self._invalid_action_count += 1

        # ----------------------------------------------------------------
        # Compute target for reward shaping
        # ----------------------------------------------------------------
        target_pos = self._current_target()

        # ----------------------------------------------------------------
        # All delivered?
        # ----------------------------------------------------------------
        all_delivered = self._delivered_items >= len(self._items)

        reward = compute_reward(
            prev_robot_pos=(float(prev_pos[0]), float(prev_pos[1])),
            curr_robot_pos=(float(self._robot_pos[0]), float(self._robot_pos[1])),
            target_pos=(
                (float(target_pos[0]), float(target_pos[1]))
                if target_pos is not None
                else None
            ),
            collision=collision,
            invalid_action=invalid_action,
            picked_up=picked_up,
            delivered=delivered,
            battery_level=self._battery,
            all_delivered=all_delivered,
            steps_elapsed=self._step_count,
            par_steps=self.cfg.par_steps,
        )

        self._cumulative_reward += reward.total

        # ----------------------------------------------------------------
        # Termination conditions
        # ----------------------------------------------------------------
        done = (
            all_delivered
            or self._step_count >= self.cfg.max_steps
            or self._battery_depleted
        )
        self._done = done

        obs = self._build_observation()

        info: Dict = {
            "delivered_items": self._delivered_items,
            "total_items": len(self._items),
            "collision_count": self._collision_count,
            "invalid_action_count": self._invalid_action_count,
            "battery_level": self._battery,
            "battery_depleted": self._battery_depleted,
            "cumulative_reward": self._cumulative_reward,
            "step": self._step_count,
        }

        return obs, reward, done, info

    def state(self) -> InternalState:
        """Return the full internal state snapshot."""
        return InternalState(
            observation=self._build_observation(),
            total_items=len(self._items),
            delivered_items=self._delivered_items,
            collision_count=self._collision_count,
            invalid_action_count=self._invalid_action_count,
            cumulative_reward=self._cumulative_reward,
            done=self._done,
            task_name=self.task_name,
            seed=self.base_seed,
        )

    def grade(self) -> float:
        """Return the deterministic episode score in [0, 1]."""
        return grade_episode(
            task_name=self.task_name,
            total_items=len(self._items),
            delivered_items=self._delivered_items,
            collision_count=self._collision_count,
            invalid_action_count=self._invalid_action_count,
            steps_taken=self._step_count,
            battery_depleted=self._battery_depleted,
        )

    # ------------------------------------------------------------------
    # Internal helpers – placement
    # ------------------------------------------------------------------

    def _is_free(self, r: int, c: int) -> bool:
        """True if cell (r, c) is walkable and within bounds."""
        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        return self._grid[r][c] == 0

    def _has_free_neighbour(self, r: int, c: int) -> bool:
        """Return True if (r, c) has at least one free orthogonal neighbour."""
        for dr, dc in _HEADING_DELTA.values():
            nr, nc = r + dr, c + dc
            if self._is_free(nr, nc) and (nr, nc) != self._robot_pos:
                return True
        return False

    def _random_free_cell(self, require_neighbour: bool = False) -> Tuple[int, int]:
        """
        Return a random free (0) cell, excluding robot position.
        When require_neighbour=True the chosen cell must also have at least
        one free orthogonal neighbour so the robot can navigate into / out of it.
        """
        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
        attempts = 0
        while attempts < 10_000:
            r = self._rng.randint(1, rows - 2)
            c = self._rng.randint(1, cols - 2)
            if self._grid[r][c] == 0 and (r, c) != self._robot_pos:
                if require_neighbour and not self._has_free_neighbour(r, c):
                    attempts += 1
                    continue
                return (r, c)
            attempts += 1
        raise RuntimeError("Could not place entity – grid too cluttered.")

    def _place_stations(self) -> List[StationInfo]:
        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
        stations = []
        # Prefer bottom edge area for stations
        edge_candidates = [
            (rows - 2, c) for c in range(1, cols - 1)
        ]
        self._rng.shuffle(edge_candidates)
        for i in range(self.cfg.num_stations):
            pos = edge_candidates[i]
            stations.append(StationInfo(station_id=i, position=pos))
        return stations

    def _place_charging_stations(self) -> List[Tuple[int, int]]:
        rows = self.cfg.grid_rows
        spots: List[Tuple[int, int]] = []
        # Place near top-right corners
        for i in range(self.cfg.num_charging_stations):
            r = 1 + i
            c = self.cfg.grid_cols - 2
            self._grid[r][c] = 0   # clear any shelf
            spots.append((r, c))
        return spots

    def _place_static_obstacles(self) -> None:
        placed = 0
        attempts = 0
        while placed < self.cfg.num_static_obstacles and attempts < 50_000:
            r, c = self._random_free_cell()
            # Don't block charging or delivery station cells
            occupied = any(
                st.position == (r, c) for st in self._stations
            ) or (r, c) in self._charging_stations
            if not occupied:
                self._grid[r][c] = 1
                placed += 1
            attempts += 1

    def _place_items(self) -> List[ItemInfo]:
        items: List[ItemInfo] = []
        num_stations = len(self._stations)
        placed_positions = set()
        for i in range(self.cfg.num_items):
            pos = self._random_free_cell(require_neighbour=True)
            # Avoid duplicates
            attempts = 0
            while pos in placed_positions and attempts < 1000:
                pos = self._random_free_cell(require_neighbour=True)
                attempts += 1
            placed_positions.add(pos)
            target_station = i % num_stations
            items.append(
                ItemInfo(
                    item_id=i,
                    position=pos,
                    is_picked=False,
                    target_station=target_station,
                )
            )
        return items

    def _update_dynamic_on_grid(self) -> None:
        """Refresh grid cells for dynamic obstacles."""
        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
        # Clear previous dynamic markers
        for r in range(rows):
            for c in range(cols):
                if self._grid[r][c] == 4:
                    self._grid[r][c] = 0
        # Mark current positions
        for pos in self._dynamic_obs:
            r, c = pos
            if self._grid[r][c] == 0:
                self._grid[r][c] = 4

    def _move_dynamic_obstacles(self) -> None:
        """Move each dynamic obstacle one step in a random valid direction."""
        new_positions: List[Tuple[int, int]] = []
        for pos in self._dynamic_obs:
            r, c = pos
            candidates = [(r + dr, c + dc) for dr, dc in _DIRECTIONS]
            valid = [
                (nr, nc)
                for nr, nc in candidates
                if self._is_free(nr, nc) and (nr, nc) != self._robot_pos
            ]
            if valid:
                new_pos = self._rng.choice(valid)
            else:
                new_pos = pos
            new_positions.append(new_pos)
        self._dynamic_obs = new_positions
        self._update_dynamic_on_grid()

    # ------------------------------------------------------------------
    # Internal helpers – actions
    # ------------------------------------------------------------------

    def _try_move_forward(self) -> bool:
        """
        Attempt to move one cell in the current heading direction.
        Returns True if a collision occurred (move blocked).
        """
        dr, dc = _HEADING_DELTA[self._robot_heading]
        nr = self._robot_pos[0] + dr
        nc = self._robot_pos[1] + dc
        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols

        # Out of bounds
        if not (0 <= nr < rows and 0 <= nc < cols):
            return True   # collision with boundary

        cell = self._grid[nr][nc]
        if cell in (1, 4):   # shelf / dynamic obstacle
            return True

        # Valid move
        self._robot_pos = (nr, nc)
        return False

    def _try_pick(self) -> bool:
        """
        Attempt to pick an item at the current cell.
        Returns True on success.
        """
        if self._carrying:
            return False   # already holding something
        for item in self._items:
            if not item.is_picked and item.position == self._robot_pos:
                item.is_picked = True
                self._carrying = True
                self._carried_item_id = item.item_id
                return True
        return False

    def _try_drop(self) -> bool:
        """
        Attempt to drop the carried item at a delivery station.
        Returns True if drop was a valid delivery.
        """
        if not self._carrying:
            return False
        # Find the carried item
        item = next(
            (it for it in self._items if it.item_id == self._carried_item_id),
            None,
        )
        if item is None:
            return False
        # Check we are at the correct station
        target_station = self._stations[item.target_station]
        if self._robot_pos == target_station.position:
            self._carrying = False
            self._carried_item_id = None
            return True
        # Standing on a station but wrong one → invalid
        return False

    def _try_recharge(self) -> bool:
        """Recharge battery if on a charging station. Returns True on success."""
        if self._robot_pos in self._charging_stations:
            self._battery = min(100.0, self._battery + 30.0)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers – observation
    # ------------------------------------------------------------------

    def _current_target(self) -> Optional[Tuple[int, int]]:
        """
        Return the current navigation waypoint.
        If carrying an item → target is the delivery station.
        Otherwise → target is the nearest unpicked item.
        """
        if self._carrying and self._carried_item_id is not None:
            item = next(
                (it for it in self._items if it.item_id == self._carried_item_id),
                None,
            )
            if item is not None:
                station = self._stations[item.target_station]
                return station.position
        # Navigate to nearest unpicked item
        unpicked = [it for it in self._items if not it.is_picked]
        if not unpicked:
            return None
        rr, rc = self._robot_pos
        nearest = min(
            unpicked,
            key=lambda it: abs(it.position[0] - rr) + abs(it.position[1] - rc),
        )
        return nearest.position

    def _nearby_obstacles(self) -> List[ObstacleInfo]:
        """Return obstacles within sensor_range of the robot."""
        rr, rc = self._robot_pos
        sr = self.cfg.sensor_range
        obstacles: List[ObstacleInfo] = []
        for r in range(max(0, rr - sr), min(self.cfg.grid_rows, rr + sr + 1)):
            for c in range(max(0, rc - sr), min(self.cfg.grid_cols, rc + sr + 1)):
                cell = self._grid[r][c]
                if cell == 1:
                    obstacles.append(ObstacleInfo(position=(r, c), is_dynamic=False))
                elif cell == 4:
                    obstacles.append(ObstacleInfo(position=(r, c), is_dynamic=True))
        return obstacles

    def _partial_grid(self) -> List[List[int]]:
        """
        Return the full grid (with dynamic obstacles reflected).
        In a production system this could be a partial view.
        """
        return deepcopy(self._grid)

    def _build_observation(self) -> Observation:
        target = self._current_target()
        robot_state = RobotState(
            position=(float(self._robot_pos[0]), float(self._robot_pos[1])),
            velocity=(float(_HEADING_DELTA[self._robot_heading][0]), float(_HEADING_DELTA[self._robot_heading][1])),   # unit velocity in heading direction
            heading=self._robot_heading,
            battery_level=round(self._battery, 2),
            carrying_item=self._carrying,
            carried_item_id=self._carried_item_id,
        )
        return Observation(
            robot=robot_state,
            grid_map=self._partial_grid(),
            nearby_obstacles=self._nearby_obstacles(),
            item_locations=deepcopy(self._items),
            stations=deepcopy(self._stations),
            current_target=(
                (int(target[0]), int(target[1])) if target is not None else None
            ),
            steps_remaining=self.cfg.max_steps - self._step_count,
            episode_time=self._step_count,
        )