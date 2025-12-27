FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
# Explicitly mark we're inside Docker for run_app.py
ENV RUN_IN_DOCKER=1

# Expose API port
EXPOSE 8000

# Run the orchestrator script which:
# 1) cleans data/artifacts/mlflow/mlrun
# 2) runs data download & preprocessing
# 3) trains the model
# 4) starts uvicorn for src.api.main:app
CMD ["python", "run_app.py"]