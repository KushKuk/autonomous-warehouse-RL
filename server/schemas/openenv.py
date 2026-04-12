"""Pydantic request/response models for the HTTP OpenEnv surface."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    task_name: str = Field(default="easy", description="Task difficulty to load")
    seed: int = Field(default=42, description="Episode seed")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional existing session identifier to replace",
    )


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Active environment session id")
    action_type: int = Field(..., ge=0, le=5, description="Discrete action id")


class StateRequest(BaseModel):
    session_id: str = Field(..., description="Active environment session id")


class ResetResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    info: Dict[str, Any]
    state: Optional[Dict[str, Any]] = None
    task: Optional[str] = None


class StepResponse(BaseModel):
    session_id: str
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
    grade: Optional[float] = None
    state: Optional[Dict[str, Any]] = None
    reward_total: Optional[float] = None


class StateResponse(BaseModel):
    session_id: str
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
