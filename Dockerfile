# CAI (Cybersecurity AI) v1.1.5 — build from this source (fork VektraShift/cai@cai-1.1.5-release).
# Multi-stage: build the wheel, then a slim runtime. No official upstream image exists.
# The FastAPI OpenAPI fix is applied on the fly (inline sed). OpenTelemetry/Phoenix tracing deps
# are regular pyproject dependencies, so no build-time tracing patch is required.
#
# Build (from repo root):
#   # CHANGE `vektrashift` to your Docker Hub owner
#   docker buildx build --platform linux/amd64 -t vektrashift/cai:v1.1.5-otel \
#     -f Dockerfile --push .

# ---- Build stage ----
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

# Git is needed to clone the pinned branch (or build from this checkout).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch cai-1.1.5-release \
    https://github.com/VektraShift/cai.git /src/cai

WORKDIR /src/cai

# FastAPI OpenAPI fix: `Dict[str, List[str]]` with `from __future__ import annotations` becomes a
# string ForwardRef pydantic can't resolve -> /api/openapi.json 500. Use built-in generics.
RUN sed -i 's/-> Dict\[str, List\[str\]\]:/-> dict[str, list[str]]:/g' \
    src/cai/api/app.py \
    && grep -n "dict\[str, list\[str\]\]" src/cai/api/app.py

# Build + install the package (pyproject now includes opentelemetry-* + openinference deps).
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel hatchling \
    && pip install . \
    && find /src/cai -maxdepth 1 -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CAI_LICENSE_OFF=1

# CAI tools (fetch_url) may need ca-certificates; curl for parity.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the build stage (no repo source in the final image).
COPY --from=build /opt/venv /opt/venv

# CAI writes config/memory under a writable dir.
ENV HOME=/home/cai
RUN useradd -m -u 1000 cai \
    && mkdir -p /home/cai/.cai \
    && chown -R cai:cai /home/cai
USER cai
WORKDIR /home/cai

EXPOSE 8080

# Default: run the CAI FastAPI backend on 0.0.0.0:8080 (LAN ingress hosts it).
ENTRYPOINT ["cai"]
CMD ["--api", "--api-host", "0.0.0.0", "--api-port", "8080"]
