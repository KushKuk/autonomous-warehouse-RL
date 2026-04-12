"""FastAPI application object exposed for OpenEnv runtime hosting."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

from server.apis.openenv import router as openenv_router
from server.apis.system import router as system_router
from server.handlers.errors import register_exception_handlers
from server.middleware.request_context import register_middleware
from server.services.runtime_service import runtime_service
from env.tasks import TASKS


app = FastAPI(
    title="Autonomous Warehouse Robot",
    version="1.0.0",
    description="FastAPI-hosted OpenEnv runtime for the warehouse simulator.",
)

register_middleware(app)
register_exception_handlers(app)
app.include_router(system_router)
app.include_router(openenv_router)


_HOME_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Autonomous Warehouse Robot</title>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: #fffaf0;
      --ink: #1a1a1a;
      --muted: #6b665e;
      --line: #d6ccb9;
      --accent: #006d77;
      --accent-2: #d97706;
      --good: #2a9d8f;
      --bad: #c1121f;
      --free: #f8f5ef;
      --wall: #6b705c;
      --station: #ffb703;
      --charge: #219ebc;
      --obstacle: #bc6c25;
      --robot: #d00000;
      --item: #2a9d8f;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(255,183,3,0.18), transparent 25%),
        linear-gradient(180deg, #f8f3e8 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
      margin-bottom: 20px;
    }
    .panel {
      background: rgba(255, 250, 240, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(43, 37, 25, 0.08);
      padding: 20px;
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 {
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
      margin-bottom: 12px;
    }
    .sub {
      color: var(--muted);
      font-size: 1rem;
      max-width: 46rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    .field {
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }
    label {
      font-size: 0.9rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    input, select, button {
      font: inherit;
    }
    input, select {
      width: 100%;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      color: var(--ink);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .btn-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 12px 14px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button.secondary { background: var(--accent-2); }
    button.ghost {
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    button:hover { transform: translateY(-1px); opacity: 0.95; }
    .actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0 8px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.65);
    }
    .stat .k {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat .v {
      font-size: 1.25rem;
      margin-top: 6px;
    }
    .viz {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 20px;
      align-items: start;
    }
    #grid {
      display: grid;
      gap: 3px;
      align-content: start;
      min-height: 360px;
    }
    .cell {
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 8px;
      border: 1px solid rgba(0,0,0,0.08);
      position: relative;
      overflow: hidden;
    }
    .cell::after {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      font-size: 0.9rem;
      font-weight: 700;
    }
    .legend {
      display: grid;
      gap: 10px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
    }
    .swatch {
      width: 16px;
      height: 16px;
      border-radius: 5px;
      border: 1px solid rgba(0,0,0,0.12);
    }
    .log {
      margin-top: 18px;
      background: #231f20;
      color: #f7f3e9;
      border-radius: 14px;
      padding: 14px;
      min-height: 140px;
      font-family: "Courier New", monospace;
      white-space: pre-wrap;
    }
    .status { color: var(--muted); }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    @media (max-width: 980px) {
      .hero, .grid, .viz { grid-template-columns: 1fr; }
      .actions, .btn-row, .row, .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <section class="panel">
        <p class="status">Warehouse Runtime Control Surface</p>
        <h1>Run the robot, inspect the grid, and keep the API intact.</h1>
        <p class="sub">The API key field is browser-side only. It is stored in local storage for your convenience and never sent to the server unless you explicitly build that into your own client flow.</p>
      </section>
      <section class="panel">
        <div class="field">
          <label for="apiKey">API Key</label>
          <input id="apiKey" type="password" placeholder="sk-... or HF token" />
        </div>
        <div class="row">
          <div class="field">
            <label for="taskName">Task</label>
            <select id="taskName">
              <option value="easy">easy</option>
              <option value="medium">medium</option>
              <option value="hard">hard</option>
            </select>
          </div>
          <div class="field">
            <label for="seed">Seed</label>
            <input id="seed" type="number" value="42" />
          </div>
          <div class="field">
            <label for="sessionId">Session</label>
            <input id="sessionId" type="text" placeholder="created on reset" />
          </div>
        </div>
        <div class="btn-row">
          <button id="resetBtn">Reset</button>
          <button id="refreshBtn" class="secondary">Refresh State</button>
          <button id="clearBtn" class="ghost">Clear Log</button>
        </div>
      </section>
    </div>

    <section class="panel">
      <h2>Live State</h2>
      <div class="stat-grid">
        <div class="stat"><div class="k">Position</div><div class="v" id="robotPos">-</div></div>
        <div class="stat"><div class="k">Heading</div><div class="v" id="robotHeading">-</div></div>
        <div class="stat"><div class="k">Battery</div><div class="v" id="battery">-</div></div>
        <div class="stat"><div class="k">Target</div><div class="v" id="currentTarget">-</div></div>
      </div>
      <div class="viz">
        <div id="grid"></div>
        <div class="legend">
          <div class="legend-item"><span class="swatch" style="background: var(--free)"></span> Free floor</div>
          <div class="legend-item"><span class="swatch" style="background: var(--wall)"></span> Wall or shelf</div>
          <div class="legend-item"><span class="swatch" style="background: var(--station)"></span> Delivery station</div>
          <div class="legend-item"><span class="swatch" style="background: var(--charge)"></span> Charging station</div>
          <div class="legend-item"><span class="swatch" style="background: var(--obstacle)"></span> Dynamic obstacle</div>
          <div class="legend-item"><span class="swatch" style="background: var(--robot)"></span> Robot</div>
          <div class="legend-item"><span class="swatch" style="background: var(--item)"></span> Item</div>
          <div class="panel" style="padding:14px;">
            <h3>Moves</h3>
            <div class="actions">
              <button data-action="0">Move Forward</button>
              <button data-action="1">Turn Left</button>
              <button data-action="2">Turn Right</button>
              <button data-action="3">Pick Item</button>
              <button data-action="4">Drop Item</button>
              <button data-action="5">Recharge</button>
            </div>
          </div>
        </div>
      </div>
      <div class="log" id="log">Waiting for reset...</div>
    </section>
  </div>

  <script>
    const base = "";
    const apiKeyInput = document.getElementById("apiKey");
    const taskNameInput = document.getElementById("taskName");
    const seedInput = document.getElementById("seed");
    const sessionIdInput = document.getElementById("sessionId");
    const logEl = document.getElementById("log");
    const gridEl = document.getElementById("grid");
    const robotPosEl = document.getElementById("robotPos");
    const robotHeadingEl = document.getElementById("robotHeading");
    const batteryEl = document.getElementById("battery");
    const currentTargetEl = document.getElementById("currentTarget");

    const cellColors = {
      0: "var(--free)",
      1: "var(--wall)",
      2: "var(--station)",
      3: "var(--charge)",
      4: "var(--obstacle)"
    };
    const headingNames = ["NORTH", "EAST", "SOUTH", "WEST"];

    apiKeyInput.value = localStorage.getItem("warehouse_api_key") || "";
    apiKeyInput.addEventListener("input", () => {
      localStorage.setItem("warehouse_api_key", apiKeyInput.value);
    });

    function appendLog(line, kind = "status") {
      const stamp = new Date().toLocaleTimeString();
      logEl.innerHTML += "\\n[" + stamp + "] " + line;
      logEl.className = "log " + kind;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function normalizeObservation(payload) {
      if (!payload) return null;
      if (payload.observation) return payload.observation;
      if (payload.state && payload.state.observation) return payload.state.observation;
      if (payload.state && payload.state.grid_map) return payload.state;
      return null;
    }

    function renderGrid(observation) {
      gridEl.innerHTML = "";
      if (!observation || !observation.grid_map) {
        gridEl.textContent = "No grid loaded.";
        return;
      }
      const grid = observation.grid_map;
      const rows = grid.length;
      const cols = rows ? grid[0].length : 0;
      gridEl.style.gridTemplateColumns = `repeat(${cols}, minmax(22px, 1fr))`;

      const robotPos = observation.robot.position.map((v) => Math.round(v));
      const itemMap = new Map();
      for (const item of observation.item_locations || []) {
        if (!item.is_picked) {
          itemMap.set(item.position.join(","), item);
        }
      }

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell = document.createElement("div");
          cell.className = "cell";
          cell.style.background = cellColors[grid[r][c]] || "var(--free)";

          const key = [r, c].join(",");
          if (robotPos[0] === r && robotPos[1] === c) {
            cell.style.background = "var(--robot)";
            cell.style.color = "#fff";
            cell.textContent = "R";
          } else if (itemMap.has(key)) {
            cell.style.background = "var(--item)";
            cell.style.color = "#fff";
            cell.textContent = "I";
          }
          gridEl.appendChild(cell);
        }
      }
    }

    function updateDashboard(payload) {
      const observation = normalizeObservation(payload);
      if (!observation) return;

      renderGrid(observation);
      const robot = observation.robot || {};
      const pos = robot.position || ["?", "?"];
      const heading = Number.isInteger(robot.heading) ? headingNames[robot.heading] : "-";
      robotPosEl.textContent = pos.join(", ");
      robotHeadingEl.textContent = heading;
      batteryEl.textContent = robot.battery_level != null ? `${robot.battery_level.toFixed(1)}%` : "-";
      currentTargetEl.textContent = observation.current_target ? observation.current_target.join(", ") : "None";
    }

    async function callApi(path, body) {
      const res = await fetch(base + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {})
      });
      const text = await res.text();
      let payload = null;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { raw: text };
      }
      if (!res.ok) {
        throw new Error(`${res.status} ${JSON.stringify(payload)}`);
      }
      return payload;
    }

    async function doReset() {
      const payload = await callApi("/reset", {
        task_name: taskNameInput.value,
        task: taskNameInput.value,
        seed: Number(seedInput.value || 42)
      });
      sessionIdInput.value = payload.session_id || "";
      updateDashboard(payload);
      appendLog(`reset -> session ${payload.session_id}`, "good");
    }

    async function doStep(actionType) {
      if (!sessionIdInput.value) {
        appendLog("reset the environment first", "bad");
        return;
      }
      const payload = await callApi("/step", {
        session_id: sessionIdInput.value,
        action_type: Number(actionType)
      });
      updateDashboard(payload);
      appendLog(`step -> action ${actionType}, done=${payload.done}, reward=${JSON.stringify(payload.reward)}`, payload.done ? "good" : "status");
    }

    async function doRefresh() {
      const payload = sessionIdInput.value
        ? await callApi("/state", { session_id: sessionIdInput.value })
        : await (await fetch("/state")).json();
      updateDashboard(payload);
      appendLog("state refreshed", "status");
    }

    document.getElementById("resetBtn").addEventListener("click", () => doReset().catch((err) => appendLog(err.message, "bad")));
    document.getElementById("refreshBtn").addEventListener("click", () => doRefresh().catch((err) => appendLog(err.message, "bad")));
    document.getElementById("clearBtn").addEventListener("click", () => { logEl.textContent = "Log cleared."; logEl.className = "log"; });
    document.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => doStep(button.dataset.action).catch((err) => appendLog(err.message, "bad")));
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_HOME_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks")
def tasks() -> dict[str, list[str]]:
    return {"tasks": runtime_service.list_tasks()}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    return {
        "name": "autonomous-warehouse-robot",
        "description": "Warehouse robot environment exposed through a FastAPI OpenEnv runtime.",
        "version": "1.0.0",
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": {
            "type": "integer",
            "enum": list(range(6)),
            "names": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "PICK_ITEM", "DROP_ITEM", "RECHARGE"],
        },
        "tasks": list(TASKS.keys()),
        "observation": "structured warehouse observation",
        "state": "structured warehouse internal state",
    }


@app.post("/mcp")
async def mcp(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {"jsonrpc": "2.0", "id": body.get("id", 1), "result": {}}


@app.get("/state")
def compatibility_state() -> dict[str, Any]:
    session_id = runtime_service.latest_session_id()
    if session_id is None:
        raise KeyError("Unknown session_id. Call /reset first.")
    return runtime_service.state(session_id=session_id)


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
