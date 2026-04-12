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


@router.post("/reset", response_model=ResetResponse)
async def reset_environment(request: Request) -> ResetResponse:
    payload = await _read_json_payload(request)
    body = ResetRequest(**payload)
    payload = runtime_service.reset(
        task_name=body.task_name,
        seed=body.seed,
        session_id=body.session_id,
    )
    return ResetResponse(**payload)


@router.post("/step", response_model=StepResponse)
async def step_environment(request: Request) -> StepResponse:
    payload = await _read_json_payload(request)
    body = StepRequest(**payload)
    payload = runtime_service.step(
        session_id=body.session_id,
        action_type=body.action_type,
    )
    return StepResponse(**payload)


@router.post("/state", response_model=StateResponse)
async def get_environment_state(request: Request) -> StateResponse:
    payload = await _read_json_payload(request)
    body = StateRequest(**payload)
    payload = runtime_service.state(session_id=body.session_id)
    return StateResponse(**payload)
