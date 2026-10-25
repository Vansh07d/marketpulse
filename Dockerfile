# syntax=docker/dockerfile:1.7
# One Dockerfile, one requirements.txt, two images: dashboard and worker (worker adds CPU torch).
ARG PYTHON_VERSION=3.12

# ---------- shared base ----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# create the HF cache dir up front so the named volume mounted there inherits the app user's ownership
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.hf_cache \
    && chown -R app:app /app

# ---------- dependencies (installed once, shared by both images) ----------
FROM base AS deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---------- dashboard ----------
FROM deps AS dashboard
COPY --chown=app:app *.py ./
USER app
EXPOSE 8503
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8503/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app2.py", "--server.address=0.0.0.0", "--server.port=8503", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# ---------- worker ----------
FROM deps AS worker
# CPU wheel by default (small). For an NVIDIA GPU build: TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${TORCH_INDEX}" torch
COPY --chown=app:app *.py ./
USER app
ENV HF_HOME=/app/.hf_cache
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; from redis_client import get_redis; sys.exit(0 if get_redis().ping() else 1)"
CMD ["python", "realtime_worker.py"]