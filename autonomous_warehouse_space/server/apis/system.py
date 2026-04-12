"""System metadata routes."""

from __future__ import annotations

from fastapi import APIRouter

from server.schemas.openenv import HealthResponse
from server.services.runtime_service import runtime_service


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": runtime_service.list_tasks()}
