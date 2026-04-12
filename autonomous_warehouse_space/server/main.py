"""Local development entrypoint for running the FastAPI server."""

from __future__ import annotations

import os

import uvicorn

from server.app import app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
