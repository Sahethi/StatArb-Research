# syntax=docker/dockerfile:1.6

# ── Stage 1: build the Vite frontend ────────────────────────────────────────
FROM node:20-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime serving FastAPI + built frontend ────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install -r requirements.txt -r backend-requirements.txt

COPY config.py ./
COPY statarb/ ./statarb/
COPY backend/ ./backend/

COPY --from=frontend /frontend/dist ./frontend/dist

ENV STATARB_FRONTEND_DIST=/app/frontend/dist \
    STATARB_CACHE_DIR=/app/.cache/api \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
