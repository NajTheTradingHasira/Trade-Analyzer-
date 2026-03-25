# ── Stage 1: build dependencies ─────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

# Default: run the alerts pipeline daily (Slack-only, high-conviction signals)
CMD ["python", "patches/alerts.py", "--slack-only", "--min-score", "7"]
