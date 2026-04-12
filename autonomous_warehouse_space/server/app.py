"""FastAPI application object exposed for OpenEnv runtime hosting."""

from __future__ import annotations

from fastapi import FastAPI

from server.apis.openenv import router as openenv_router
from server.apis.system import router as system_router
from server.handlers.errors import register_exception_handlers
from server.middleware.request_context import register_middleware


app = FastAPI(
    title="Autonomous Warehouse Robot",
    version="1.0.0",
    description="FastAPI-hosted OpenEnv runtime for the warehouse simulator.",
)

register_middleware(app)
register_exception_handlers(app)
app.include_router(system_router)
app.include_router(openenv_router)
