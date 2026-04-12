"""Request tracing middleware."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        started = time.time()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["X-Process-Time"] = f"{time.time() - started:.6f}"
        return response
