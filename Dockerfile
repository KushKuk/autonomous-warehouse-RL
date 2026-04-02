# ──────────────────────────────────────────────────────────────────────────────
# Autonomous Warehouse Robot – Docker image
# Base: python:3.10-slim
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Metadata
LABEL maintainer="OpenEnv Warehouse Team"
LABEL description="Autonomous Warehouse Robot RL environment + baseline"
LABEL version="1.0.0"

# Prevent Python from buffering stdout/stderr (useful for live log streaming)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install OS-level dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first (layer caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY env/       ./env/
COPY baseline/  ./baseline/
COPY openenv.yaml ./

# Default command: run the baseline evaluation script
# Requires OPENAI_API_KEY to be passed at runtime, e.g.:
#   docker run -e OPENAI_API_KEY=sk-... warehouse-robot
CMD ["python", "baseline/run_agent.py"]