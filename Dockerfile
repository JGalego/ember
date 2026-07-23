FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by matplotlib / networkx wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch from a specific, slightly older CUDA build *before* the rest
# of requirements.txt. Plain `pip install torch` pulls whatever CUDA runtime
# PyPI currently bundles by default, which drifts forward over time (e.g.
# CUDA 13.0 as of writing) and can be newer than a rented GPU host's NVIDIA
# driver actually supports -- that fails at runtime with "The NVIDIA driver
# on your system is too old" even though the GPU itself is fine. cu128
# (CUDA 12.8) is broadly supported by current cloud GPU hosts; if this index
# stops publishing wheels for the current torch release in the future, check
# https://pytorch.org/get-started/locally/ for the current recommended tag.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

EXPOSE 8000

# Default: serve the API. Override the command to run training/benchmarks instead, e.g.:
#   docker run ember python scripts/train.py dataset=sudoku
CMD ["uvicorn", "ember.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
