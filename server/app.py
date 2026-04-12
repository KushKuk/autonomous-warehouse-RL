"""FastAPI application object exposed for OpenEnv runtime hosting."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request

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


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Autonomous Warehouse Robot API"}


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
