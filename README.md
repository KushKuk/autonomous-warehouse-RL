# 🤖 Autonomous Warehouse Robot – OpenEnv Environment

A production-grade, OpenEnv-compliant reinforcement learning environment that
simulates a real-world autonomous warehouse robot navigating a 2-D grid,
picking items from shelves, avoiding obstacles, and delivering goods to
designated stations.

---

## 🏭 Real-World Relevance

Modern fulfilment centres (Amazon, Ocado, DHL) deploy fleets of autonomous
mobile robots (AMRs) to pick, transport, and sort inventory. Key challenges
replicated in this environment include:

| Real-world challenge | Simulated as |
|----------------------|--------------|
| Warehouse floor plan | 2-D grid with shelf columns & aisle walkways |
| Fork-lift / conveyor conflicts | Dynamic obstacles with random walk |
| Battery / charge management | Battery drain + charging stations |
| Order picking workflow | Item pickup → delivery station routing |
| Collision avoidance | Negative reward + blocked moves |
| Route optimisation | Step penalty + efficiency bonus vs par steps |

---

## 📐 Environment Structure

```
warehouse_env/
├── env/
│   ├── __init__.py        – package exports
│   ├── environment.py     – WarehouseEnv (OpenEnv API)
│   ├── models.py          – Pydantic: Observation, Action, Reward, InternalState
│   ├── rewards.py         – Dense, decomposed reward function
│   └── tasks.py           – Task configs (easy/medium/hard) + graders
├── baseline/
│   └── run_agent.py       – LLM agent via OpenAI API
├── openenv.yaml           – OpenEnv spec manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🗺️ Observation Space

The `Observation` model (Pydantic) contains:

| Field | Type | Description |
|-------|------|-------------|
| `robot.position` | `(float, float)` | Current (row, col) in grid coordinates |
| `robot.velocity` | `(float, float)` | Unit velocity in heading direction |
| `robot.heading` | `int` | 0=North, 1=East, 2=South, 3=West |
| `robot.battery_level` | `float [0,100]` | Remaining battery percentage |
| `robot.carrying_item` | `bool` | Whether robot holds an item |
| `robot.carried_item_id` | `int \| None` | ID of held item |
| `grid_map` | `List[List[int]]` | Full grid: 0=free, 1=shelf, 2=station, 3=charger, 4=dyn. obstacle |
| `nearby_obstacles` | `List[ObstacleInfo]` | Obstacles within sensor range |
| `item_locations` | `List[ItemInfo]` | All items with positions and pick state |
| `stations` | `List[StationInfo]` | All delivery stations |
| `current_target` | `(float, float) \| None` | Active navigation waypoint |
| `steps_remaining` | `int` | Steps left in episode |
| `episode_time` | `int` | Steps elapsed since reset |

---

## 🎮 Action Space

Discrete action space with 6 actions:

| ID | Name | Effect |
|----|------|--------|
| 0 | `MOVE_FORWARD` | Move 1 cell in heading direction (blocked by walls/obstacles) |
| 1 | `TURN_LEFT` | Rotate 90° counter-clockwise |
| 2 | `TURN_RIGHT` | Rotate 90° clockwise |
| 3 | `PICK_ITEM` | Pick item at current cell (must be item present & not carrying) |
| 4 | `DROP_ITEM` | Deliver carried item to correct station |
| 5 | `RECHARGE` | Recharge battery at charging station (+30% per activation) |

---

## 💰 Reward Design

Dense, decomposed reward signal — every component is logged separately:

| Component | Value | Trigger |
|-----------|-------|---------|
| Step penalty | −0.01 | Every timestep |
| Approach shaping | ±0.05 × Δdist | Moving toward/away from current target |
| Pickup reward | +2.0 | Successful item pickup |
| Delivery reward | +5.0 | Item delivered to correct station |
| Collision penalty | −1.0 | Hitting wall or obstacle |
| Invalid action penalty | −0.2 | Logically impossible action |
| Battery low penalty | −0.05/step | Battery < 20% |
| Efficiency bonus | +2.0 | All deliveries completed within par steps |

---

## 📋 Task Definitions

### Easy
- Grid: 8×8
- Items: 1
- Static obstacles: 3 | Dynamic: 0
- Max steps: 100 | Par: 30
- Battery: disabled
- Goal: single-item pickup and delivery

### Medium
- Grid: 12×12
- Items: 3
- Static obstacles: 10 | Dynamic: 0
- Max steps: 300 | Par: 150
- Battery: disabled
- Goal: multi-item collection with obstacle avoidance

### Hard
- Grid: 16×16
- Items: 5
- Static obstacles: 20 | Dynamic: 4 (move every 5 steps)
- Max steps: 500 | Par: 300
- Battery: **enabled** (episode ends if depleted)
- Goal: full warehouse run with time, energy, and obstacle constraints

---

## 📊 Grader

Each task uses a deterministic grader returning a score in **[0.0, 1.0]**:

```
score = 0.50 × delivery_ratio
      + 0.25 × path_efficiency
      + 0.15 × collision_score
      + 0.10 × invalid_action_score

if battery_depleted: score *= 0.80
```

Where:
- `delivery_ratio` = items delivered / total items
- `path_efficiency` = 1.0 if steps ≤ par, degrades linearly to 0 at 2× par
- `collision_score` = max(0, 1 − collisions/10)
- `invalid_action_score` = max(0, 1 − invalid_actions/20)

---

## ⚙️ Setup

### 1. Clone / unzip the project

```bash
cd warehouse_env
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Baseline

```bash
export OPENAI_API_KEY=sk-...

# Run with defaults (gpt-4o-mini, 3 episodes per task, seed=42)
python baseline/run_agent.py

# Custom options
OPENAI_MODEL=gpt-4o NUM_EPISODES=5 SEED=123 python baseline/run_agent.py
```

### Expected output

```
============================================================
  Autonomous Warehouse Robot – Baseline Evaluation
  Model : gpt-4o-mini
  Seed  : 42
  Episodes per task: 3
============================================================

────────────────────────────────────────────────────────────
  Task: EASY
────────────────────────────────────────────────────────────
  Episode 1/3  (seed=...)
  step=001 action=MOVE_FORWARD     reward=+0.040 delivered=0/1 battery=99.7%
  ...
  → Episode score: 0.8250

============================================================
  FINAL RESULTS
============================================================
  easy     : 0.7833
  medium   : 0.5417
  hard     : 0.3100
  overall  : 0.5450
============================================================
```

---

## 🐳 Docker

```bash
# Build
docker build -t warehouse-robot .

# Run baseline
docker run -e OPENAI_API_KEY=sk-... warehouse-robot

# Custom options
docker run \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-4o \
  -e NUM_EPISODES=5 \
  warehouse-robot
```

---

## 🔌 Programmatic Usage

```python
from env.environment import WarehouseEnv
from env.models import Action, ActionType

env = WarehouseEnv(task_name="medium", seed=42)
obs = env.reset()

done = False
while not done:
    # Your policy here
    action = Action(action_type=ActionType.MOVE_FORWARD)
    obs, reward, done, info = env.step(action)
    print(f"reward={reward.total:.3f} | info={info}")

print(f"Episode score: {env.grade():.4f}")

# Full internal snapshot
snapshot = env.state()
```

---

## 📄 License

MIT