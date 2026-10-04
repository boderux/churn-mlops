# syntax=docker/dockerfile:1.7
# ---------- Stage 1: build the virtualenv ----------
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

# ---------- Stage 2: minimal, non-root runtime ----------
FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app app
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="churn-api" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/your-org/churn-mlops"
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src ARTIFACT_DIR=/app/artifacts PROJECT_ROOT=/app GIT_SHA=${GIT_SHA} \
    MPLCONFIGDIR=/tmp/mpl
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
# The champion model bundle is baked in by CI/CD (artifacts/champion). Locally it is mounted.
COPY --chown=app:app artifacts ./artifacts
RUN mkdir -p /app/artifacts && chown -R app:app /app/artifacts
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
