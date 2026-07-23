FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by matplotlib / networkx wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

EXPOSE 8000

# Default: serve the API. Override the command to run training/benchmarks instead, e.g.:
#   docker run kona-ebm python scripts/train.py dataset=sudoku
CMD ["uvicorn", "kona_ebm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
