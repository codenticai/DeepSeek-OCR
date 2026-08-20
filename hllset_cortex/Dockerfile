# ─────────────────────────────────────────────────────────────────────────────
# hllset-cortex Service — DeepSeek-OCR HLLSet Pipeline
# Multi-stage: Rust build + Python runtime
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build hllset_py (Rust → Python via maturin) ───────────────────
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install maturin --break-system-packages

WORKDIR /build
COPY crates/hllset_py/ crates/hllset_py/

WORKDIR /build/crates/hllset_py
RUN maturin build --release --out /wheels

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install hllset_py wheel from builder
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Install Flask
RUN pip install --no-cache-dir flask

# Copy hllset_cortex Python package and service
COPY pyproject.toml setup.py* ./
COPY src/ src/
COPY service.py .

# Install hllset-cortex Python package
RUN pip install --no-cache-dir -e .

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9092/health || exit 1

EXPOSE 9092

CMD ["python", "service.py"]
