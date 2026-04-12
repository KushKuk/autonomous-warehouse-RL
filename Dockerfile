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
COPY env/ ./env/
COPY server/ ./server/
COPY baseline/ ./baseline/
COPY client_notebooks/ ./client_notebooks/
COPY runtime_data/ ./runtime_data/
COPY app.py ./
COPY client.py ./
COPY inference.py ./
COPY interface.py ./
COPY models.py ./
COPY scenario_config.json ./
COPY openenv.yaml ./

# Default command: run the FastAPI OpenEnv server
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
