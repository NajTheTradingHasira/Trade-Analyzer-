# APEX Trade Analyzer — Docker image for Railway / local deployment
# Runs the alert pipeline by default, or any CLI mode via CMD override

# ── Stage 1: build dependencies ─────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# System deps for lxml/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 libxslt1.1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# Env vars read at runtime (set in Railway dashboard)
# SLACK_WEBHOOK_URL, SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO

# Default: run alerts pipeline
CMD ["python", "patches/alerts.py", "--slack-only", "--min-score", "7"]
