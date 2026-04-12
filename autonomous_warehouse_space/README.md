---
title: Autonomous Warehouse Robot
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Autonomous Warehouse Robot

This repository is configured for a Hugging Face `Docker` Space. The container starts the FastAPI OpenEnv server and exposes the warehouse endpoints on port `7860`.

Primary runtime:

- FastAPI app: `server.app:app`
- Container startup: `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- OpenEnv endpoints:
  - `POST /reset`
  - `POST /step`
  - `POST /state`
  - `GET /system/health`
  - `GET /system/tasks`

The hackathon submission script remains [inference.py](/home/ujjwal/autonomous_warehouse_space/inference.py).

## Required Environment Variables

For `inference.py`:

- `HF_TOKEN`
- `API_BASE_URL` optional, default is `https://api.openai.com/v1`
- `MODEL_NAME` optional, default is `gpt-4.1-mini`
- `TASK_NAME` optional, default is `easy`

## Project Structure

Core runtime files:

- `server/app.py` - FastAPI app object
- `server/apis/openenv.py` - `/reset`, `/step`, `/state`
- `server/apis/system.py` - health and task metadata
- `env/environment.py` - warehouse simulator
- `env/models.py` - typed action, observation, reward, and state models
- `env/rewards.py` - reward function
- `env/tasks.py` - task presets and deterministic grading
- `Dockerfile` - Docker Space startup
- `openenv.yaml` - OpenEnv runtime metadata
- `inference.py` - submission entrypoint

Optional support code:

- `app.py` - Gradio UI implementation
- `interface.py` - alternate Gradio launcher
- `baseline/run_agent.py` - older CLI runner
- `client.py` - local helper client

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Run the submission script locally:

```bash
export HF_TOKEN="your_token_here"
python inference.py
```

## Notes

- For Hugging Face Spaces, create the Space as `Docker`.
- Do not select `Gradio` for this repository if you want OpenEnv validators to hit `/reset`, `/step`, and `/state`.
- `HF_TOKEN` should be stored in Hugging Face Space secrets, not hardcoded in the repository.
