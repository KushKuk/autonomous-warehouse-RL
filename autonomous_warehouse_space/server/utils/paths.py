"""Shared filesystem paths used by the server."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "runtime_data"
DB_PATH = DATA_DIR / "seed_store.db"
SCENARIO_PATH = ROOT_DIR / "scenario_config.json"
