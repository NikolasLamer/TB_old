# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

# ── App setup ────────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

# ── Prod server ──────────────────────────────────────────────────────────────
# Two gunicorn workers are plenty; tweak if CPU‑bound.
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "trading_bot:app"]
