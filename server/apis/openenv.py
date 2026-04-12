"""HTTP routers for the warehouse OpenEnv surface."""

from __future__ import annotations

from fastapi import APIRouter, Request

from server.schemas.openenv import ResetRequest, ResetResponse, StateRequest, StateResponse, StepRequest, StepResponse
from server.services.runtime_service import runtime_service


router = APIRouter(tags=["openenv"])


async def _read_json_payload(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


_ACTION_NAME_TO_ID = {
    "MOVE_FORWARD": 0,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 2,
    "PICK_ITEM": 3,
    "DROP_ITEM": 4,
    "RECHARGE": 5,
}


def _normalize_task_name(payload: dict) -> str:
    value = payload.get("task_name") or payload.get("task") or payload.get("task_id") or "easy"
    return str(value)


def _normalize_session_id(payload: dict) -> str | None:
    value = payload.get("session_id")
    if value is None:
        return runtime_service.latest_session_id()
    return str(value)


def _normalize_action_type(payload: dict) -> int:
    if "action_type" in payload:
        return int(payload["action_type"])
    action = payload.get("action")
    if action is None:
        return 0
    if isinstance(action, int):
        return int(action)
    action_name = str(action).strip().upper().replace(" ", "_")
    return _ACTION_NAME_TO_ID.get(action_name, 0)


@router.post("/reset", response_model=ResetResponse)
async def reset_environment(request: Request) -> ResetResponse:
    payload = await _read_json_payload(request)
    body = ResetRequest(
        task_name=_normalize_task_name(payload),
        seed=int(payload.get("seed", 42)),
        session_id=payload.get("session_id"),
    )
    payload = runtime_service.reset(
        task_name=body.task_name,
        seed=body.seed,
        session_id=body.session_id,
    )
    return ResetResponse(**payload)


@router.post("/step", response_model=StepResponse)
async def step_environment(request: Request) -> StepResponse:
    payload = await _read_json_payload(request)
    session_id = _normalize_session_id(payload)
    if session_id is None:
        raise KeyError("Unknown session_id. Call /reset first.")
    body = StepRequest(
        session_id=session_id,
        action_type=_normalize_action_type(payload),
    )
    payload = runtime_service.step(
        session_id=body.session_id,
        action_type=body.action_type,
    )
    return StepResponse(**payload)


@router.post("/state", response_model=StateResponse)
async def get_environment_state(request: Request) -> StateResponse:
    payload = await _read_json_payload(request)
    session_id = _normalize_session_id(payload)
    if session_id is None:
        raise KeyError("Unknown session_id. Call /reset first.")
    body = StateRequest(session_id=session_id)
    payload = runtime_service.state(session_id=body.session_id)
    return StateResponse(**payload)
