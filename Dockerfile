# Dockerfile
# Delaware Hybrid RAG API (FastAPI + main.py)
# NOTE: Uses OpenAI API at runtime. Pass OPENAI_API_KEY via env.
# NOTE: vectorstore/ and prompts/ should be included in the image OR mounted at runtime.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# System deps (optional but helpful for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user (security best practice)
RUN useradd --create-home --shell /bin/bash appuser

# Install python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Make sure these exist in the image (or mount them):
# - vectorstore/faiss_index
# - prompts/**
# - api.py, main.py, prompt_manager.py, etc.

# Ensure files are owned by non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Optional: set workers to 1 for predictable resource usage in small containers
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]