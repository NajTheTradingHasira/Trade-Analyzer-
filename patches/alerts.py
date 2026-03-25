"""
APEX Alert Engine — alerts.py
===============================
Standalone CLI that fetches fresh data, runs the full stage-enriched
pipeline, filters to new signals, deduplicates against previous runs,
and pushes to Slack and/or email.

Usage:
  python patches/alerts.py                          # Run with defaults
  python patches/alerts.py --min-score 7            # Only high-conviction
  python patches/alerts.py --slack-only             # Skip email
  python patches/alerts.py --email-only             # Skip Slack
  python patches/alerts.py --dry-run                # Preview without sending
  python patches/alerts.py --watchlist custom.txt   # Custom watchlist

Env vars:
  SLACK_WEBHOOK_URL   — Slack incoming webhook
  SMTP_HOST           — SMTP server (e.g., smtp.gmail.com)
  SMTP_USER           — SMTP username / email
  SMTP_PASS           — SMTP password / app password
  ALERT_EMAIL_TO      — Recipient email address

Dedup:
  Stores seen alerts in alerts_history.json keyed by symbol+date+event_code.
  Only new signals are sent on each run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from patches.add_stage_enriched import stage_enriched_scan
from patches.add_as_alerts import detect_strength_alerts, print_strength_alerts, format_strength_alert_slack
from patches.add_rank_change_alerts import detect_rank_change_alerts, print_rank_change_alerts, format_rank_alert_slack

logger = logging.getLogger("alerts")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

# ── Config from env ──
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

HISTORY_FILE = Path(__file__).parent.parent / "alerts_history.json"
DEFAULT_WATCHLIST = Path(__file__).parent.parent / "watchlist.txt"


# ══════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ══════════════════════════════════════════════════════════════

def load_watchlist(path=None):
    """Load tickers from watchlist file, one per line."""
    wl_path = Path(path) if path else DEFAULT_WATCHLIST
    if not wl_path.exists():
        logger.error(f"Watchlist not found: {wl_path}")
        return []
    tickers = []
    with open(wl_path) as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith('#'):
                tickers.append(t)
    logger.info(f"Loaded {len(tickers)} tickers from {wl_path}")
    return tickers


def fetch_ohlcv(tickers, period='1y'):
    """Download OHLCV data for all tickers via yfinance."""
    logger.info(f"Downloading {period} data for {len(tickers)} tickers...")
    all_data = []

    # Download in chunks to avoid timeout
    chunk_size = 30
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

    for idx, chunk in enumerate(chunks):
        logger.info(f"  Chunk {idx+1}/{len(chunks)}: {len(chunk)} tickers")
        for sym in chunk:
            try:
                t = yf.Ticker(sym)
                df = t.history(period=period, interval='1d')
                if df.empty:
                    continue
                df = df.reset_index()
                df['symbol'] = sym
                df = df.rename(columns={
                    'Date': 'date', 'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                all_data.append(df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']])
            except Exception as e:
                logger.warning(f"  {sym}: {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Downloaded {len(result)} rows for {result['symbol'].nunique()} symbols")
    return result


# ══════════════════════════════════════════════════════════════
# 2. DEDUPLICATION
# ══════════════════════════════════════════════════════════════

def load_history():
    """Load alert history from JSON file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_history(history):
    """Save alert history to JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def dedup_key(row):
    """Generate dedup key from a signal row."""
    sym = row.get('symbol', '')
    date = str(row.get('date', ''))[:10]
    code = row.get('event_code', 'none')
    return f"{sym}|{date}|{code}"


def filter_new_signals(signals_df, history):
    """Remove signals already seen in previous runs."""
    new_rows = []
    for _, row in signals_df.iterrows():
        key = dedup_key(row)
        if key not in history:
            new_rows.append(row)
    if not new_rows:
        return pd.DataFrame()
    return pd.DataFrame(new_rows)


def mark_sent(signals_df, history):
    """Mark signals as sent in history."""
    for _, row in signals_df.iterrows():
        key = dedup_key(row)
        history[key] = {
            'symbol': row.get('symbol', ''),
            'date': str(row.get('date', ''))[:10],
            'event_code': row.get('event_code', ''),
            'event_score': float(row.get('event_score', 0)),
            'sent_at': datetime.utcnow().isoformat(),
        }
    return history


# ══════════════════════════════════════════════════════════════
# 3. FORMATTING
# ══════════════════════════════════════════════════════════════

def format_alert_text(row):
    """Format a single alert as plain text."""
    sym = row.get('symbol', '?')
    date = str(row.get('date', ''))[:10]
    price = row.get('close', 0)
    event = row.get('event_label', row.get('event_code', '?'))
    score = row.get('event_score', 0)
    stage = row.get('w_stage_label', '')
    direction = row.get('direction', '')
    gap_pct = row.get('gap_pct', 0)
    structure = row.get('structure', '')
    iv_regime = row.get('iv_regime', '')
    sell = row.get('sell_signal', False)

    lines = [
        f"{'SELL' if sell else 'BUY'} | {sym} @ ${price:.2f} | Score: {score}",
        f"  Event: {event}",
        f"  Stage: {stage} | Direction: {direction}",
    ]
    if gap_pct and not pd.isna(gap_pct):
        lines.append(f"  Gap: {gap_pct*100:.1f}%")
    if structure:
        lines.append(f"  Options: {structure} ({iv_regime} IV)")
    if row.get('stage_adj_reason'):
        lines.append(f"  Adj: {row['stage_adj_reason']}")

    return '\n'.join(lines)


def format_slack_block(row):
    """Format a single alert as a Slack mrkdwn block."""
    sym = row.get('symbol', '?')
    price = row.get('close', 0)
    event = row.get('event_label', row.get('event_code', '?'))
    score = row.get('event_score', 0)
    stage = row.get('w_stage_label', '')
    sell = row.get('sell_signal', False)
    structure = row.get('structure', '')
    iv_regime = row.get('iv_regime', '')
    gap_pct = row.get('gap_pct', 0)

    emoji = ':red_circle:' if sell else ':large_green_circle:'
    side = 'SELL' if sell else 'BUY'
    gap_str = f" | Gap {gap_pct*100:.1f}%" if gap_pct and not pd.isna(gap_pct) else ""

    text = (
        f"{emoji} *{side} {sym}* @ ${price:.2f} | Score *{score}*{gap_str}\n"
        f">{event}\n"
        f">{stage}"
    )
    if structure:
        text += f"\n>Options: _{structure}_ ({iv_regime} IV)"

    return text


def format_email_html(signals_df, regime_info=None):
    """Format all alerts as an HTML email body."""
    rows_html = ""
    for _, row in signals_df.iterrows():
        sym = row.get('symbol', '?')
        price = row.get('close', 0)
        event = row.get('event_label', '?')
        score = row.get('event_score', 0)
        stage = row.get('w_stage_label', '')
        sell = row.get('sell_signal', False)
        structure = row.get('structure', '')
        iv_regime = row.get('iv_regime', '')

        color = '#ff4444' if sell else '#00c853'
        side = 'SELL' if sell else 'BUY'

        rows_html += f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:8px;color:{color};font-weight:bold;">{side}</td>
            <td style="padding:8px;font-weight:bold;">{sym}</td>
            <td style="padding:8px;">${price:.2f}</td>
            <td style="padding:8px;">{score}</td>
            <td style="padding:8px;">{event}</td>
            <td style="padding:8px;">{stage}</td>
            <td style="padding:8px;">{structure}<br><small>({iv_regime} IV)</small></td>
        </tr>
        """

    regime_html = ""
    if regime_info:
        regime_html = f"""
        <div style="background:#1a1a2e;padding:12px;border-radius:6px;margin-bottom:16px;">
            <strong>Market Regime:</strong> {regime_info.get('regime', 'Unknown')} |
            SPY: ${regime_info.get('spy_price', 0)} |
            Vol 20d: {regime_info.get('volatility_20d', 0)}%
        </div>
        """

    return f"""
    <html>
    <body style="background:#0d1117;color:#e0e0e0;font-family:Consolas,monospace;padding:20px;">
        <h2 style="color:#58a6ff;">APEX Trade Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>
        {regime_html}
        <p>{len(signals_df)} new signal(s) detected</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:#161b22;color:#8b949e;">
                <th style="padding:8px;">Side</th>
                <th style="padding:8px;">Ticker</th>
                <th style="padding:8px;">Price</th>
                <th style="padding:8px;">Score</th>
                <th style="padding:8px;">Event</th>
                <th style="padding:8px;">Stage</th>
                <th style="padding:8px;">Options</th>
            </tr>
            {rows_html}
        </table>
        <p style="color:#666;font-size:11px;margin-top:20px;">
            Generated by APEX Trade Analyzer Suite
        </p>
    </body>
    </html>
    """


# ══════════════════════════════════════════════════════════════
# 4. DELIVERY
# ══════════════════════════════════════════════════════════════

def send_slack(signals_df, regime_info=None):
    """Send alerts to Slack via incoming webhook."""
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set - skipping Slack")
        return False

    # Build message
    header = f"*APEX Trade Alerts* - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    if regime_info:
        header += f"Regime: {regime_info.get('regime', '?')} | SPY ${regime_info.get('spy_price', 0)}\n"
    header += f"{len(signals_df)} new signal(s)\n\n"

    blocks = [header]
    for _, row in signals_df.head(20).iterrows():  # Cap at 20 to avoid Slack limits
        blocks.append(format_slack_block(row))

    if len(signals_df) > 20:
        blocks.append(f"\n_...and {len(signals_df) - 20} more signals_")

    payload = {"text": '\n\n'.join(blocks)}

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=15)
        if resp.status_code == 200:
            logger.info(f"Slack: sent {len(signals_df)} alerts")
            return True
        else:
            logger.error(f"Slack: HTTP {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"Slack: {e}")
        return False


def send_email(signals_df, regime_info=None):
    """Send alerts via email."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO]):
        missing = []
        if not SMTP_HOST: missing.append('SMTP_HOST')
        if not SMTP_USER: missing.append('SMTP_USER')
        if not SMTP_PASS: missing.append('SMTP_PASS')
        if not ALERT_EMAIL_TO: missing.append('ALERT_EMAIL_TO')
        logger.warning(f"Email not configured - missing: {', '.join(missing)}")
        return False

    sells = signals_df['sell_signal'].sum() if 'sell_signal' in signals_df.columns else 0
    subject = f"APEX Alerts: {len(signals_df)} signals ({sells} sells) - {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = ALERT_EMAIL_TO

    # Plain text fallback
    text_parts = [f"APEX Trade Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for _, row in signals_df.iterrows():
        text_parts.append(format_alert_text(row))
    msg.attach(MIMEText('\n\n'.join(text_parts), 'plain'))

    # HTML version
    html = format_email_html(signals_df, regime_info)
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info(f"Email: sent to {ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Email: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# 5. MAIN CLI
# ══════════════════════════════════════════════════════════════

def run_alerts(watchlist_path=None, min_score=5.0, period='1y',
               slack=True, email=True, dry_run=False,
               events_only=True, sell_only=False):
    """
    Full alert pipeline:
      1. Load watchlist
      2. Fetch fresh OHLCV data
      3. Run stage-enriched scan
      4. Filter to actionable signals
      5. Dedup against history
      6. Send via Slack and/or email
      7. Update history
    """
    start = time.time()

    print("=" * 60)
    print("  APEX ALERT ENGINE")
    print("=" * 60)
    print(f"  Min score:  {min_score}")
    print(f"  Slack:      {'ON' if slack else 'OFF'}")
    print(f"  Email:      {'ON' if email else 'OFF'}")
    print(f"  Dry run:    {'YES' if dry_run else 'NO'}")
    print("=" * 60)
    print()

    # 1. Load watchlist
    tickers = load_watchlist(watchlist_path)
    if not tickers:
        print("No tickers in watchlist. Exiting.")
        return

    # 2. Fetch data
    df = fetch_ohlcv(tickers, period=period)
    if df.empty:
        print("No data fetched. Exiting.")
        return

    # 3. Run scan
    result, regime, stages = stage_enriched_scan(df)

    # 4. Filter to actionable signals
    signals = result[result.get('event_code', pd.Series('none')) != 'none'].copy()
    signals = signals[signals['event_score'] >= min_score]

    if sell_only:
        signals = signals[signals.get('sell_signal', pd.Series(False)).fillna(False)]

    # HARD FILTER: only tradeable signals
    if 'tradeability_flag' in signals.columns:
        before = len(signals)
        signals = signals[signals['tradeability_flag'].fillna(False)]
        filtered = before - len(signals)
        if filtered > 0:
            print(f"  Tradeability gate: {filtered} non-tradeable signals filtered out")

    # Only look at recent signals (last 5 trading days)
    if 'date' in signals.columns:
        signals['date'] = pd.to_datetime(signals['date'])
        cutoff = signals['date'].max() - pd.Timedelta(days=7)
        signals = signals[signals['date'] >= cutoff]

    if signals.empty:
        print("\nNo new actionable signals above threshold.")
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")
        return

    # Sort by score
    signals = signals.sort_values('event_score', ascending=False)

    print(f"\n{len(signals)} signals above score {min_score} in last 5 trading days")

    # 4b. Strength alerts (compare to previous run)
    strength_alerts = detect_strength_alerts(result)
    print_strength_alerts(strength_alerts)

    # 4c. Rank change alerts (rolling history comparison)
    rank_alerts = detect_rank_change_alerts(result)
    print_rank_change_alerts(rank_alerts)

    # Combine all non-event alerts
    all_strength_alerts = strength_alerts + rank_alerts

    # 5. Dedup
    history = load_history()
    new_signals = filter_new_signals(signals, history)

    has_new_events = not new_signals.empty
    has_strength = len(all_strength_alerts) > 0

    if not has_new_events and not has_strength:
        print("No new event signals or strength alerts. Nothing to send.")
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")
        return

    if has_new_events:
        print(f"{len(new_signals)} NEW event signals (deduped from {len(signals)})")
    else:
        print("No new event signals (all deduped).")
        new_signals = pd.DataFrame()

    # Preview events
    if not new_signals.empty:
        display_cols = [c for c in [
            'symbol', 'date', 'close', 'event_label', 'event_score',
            'w_stage_label', 'structure', 'sell_signal'
        ] if c in new_signals.columns]
        print("\nNew event alerts:")
        preview = new_signals[display_cols].copy()
        if 'date' in preview.columns:
            preview['date'] = preview['date'].astype(str).str[:10]
        print(preview.to_string(index=False))

    if dry_run:
        print("\n[DRY RUN] Alerts not sent.")
        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")
        return

    # 6. Send
    slack_ok = False
    email_ok = False

    if slack:
        # Send event signals
        if not new_signals.empty:
            slack_ok = send_slack(new_signals, regime)

        # Send strength + rank alerts as a separate Slack message
        if all_strength_alerts and SLACK_WEBHOOK_URL:
            n_str = len(strength_alerts)
            n_rank = len(rank_alerts)
            strength_header = (
                f"*APEX Strength & Rank Alerts* - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{n_str} strength + {n_rank} rank change(s) detected\n\n"
            )
            strength_blocks = [strength_header]
            for sa in strength_alerts[:10]:
                strength_blocks.append(format_strength_alert_slack(sa))
            for ra in rank_alerts[:10]:
                strength_blocks.append(format_rank_alert_slack(ra))

            try:
                resp = requests.post(
                    SLACK_WEBHOOK_URL,
                    json={"text": '\n\n'.join(strength_blocks)},
                    timeout=15
                )
                if resp.status_code == 200:
                    logger.info(f"Slack: sent {len(strength_alerts)} strength alerts")
                    slack_ok = True
            except Exception as e:
                logger.error(f"Slack strength alerts: {e}")

    if email:
        if not new_signals.empty:
            email_ok = send_email(new_signals, regime)

    # 7. Update history
    if slack_ok or email_ok or (not slack and not email):
        if not new_signals.empty:
            history = mark_sent(new_signals, history)
            save_history(history)
            logger.info(f"History updated: {len(history)} total entries")

    # Summary
    elapsed = time.time() - start
    n_events = len(new_signals) if not new_signals.empty else 0
    n_strength = len(strength_alerts)
    n_rank = len(rank_alerts)
    print(f"\n{'=' * 60}")
    print(f"  ALERT SUMMARY")
    print(f"  Event signals:    {n_events} new")
    print(f"  Strength alerts:  {n_strength}")
    print(f"  Rank alerts:      {n_rank}")
    high_count = sum(1 for a in all_strength_alerts if a['severity'] == 'high')
    if high_count:
        print(f"  HIGH severity:    {high_count}")
    print(f"  Slack:   {'Sent' if slack_ok else 'Skipped/Failed'}")
    print(f"  Email:   {'Sent' if email_ok else 'Skipped/Failed'}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="APEX Alert Engine - scan watchlist and push new signals to Slack/email"
    )
    parser.add_argument('--watchlist', '-w', default=None,
                        help='Path to watchlist file (default: watchlist.txt)')
    parser.add_argument('--min-score', type=float, default=5.0,
                        help='Minimum event score to alert (default: 5.0)')
    parser.add_argument('--period', default='1y',
                        help='Data period to fetch (default: 1y)')
    parser.add_argument('--slack-only', action='store_true',
                        help='Only send via Slack')
    parser.add_argument('--email-only', action='store_true',
                        help='Only send via email')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview alerts without sending')
    parser.add_argument('--sell-only', action='store_true',
                        help='Only alert on sell signals')
    parser.add_argument('--reset-history', action='store_true',
                        help='Clear alert history before running')

    args = parser.parse_args()

    if args.reset_history and HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
        print("Alert history cleared.")

    slack = not args.email_only
    email = not args.slack_only

    run_alerts(
        watchlist_path=args.watchlist,
        min_score=args.min_score,
        period=args.period,
        slack=slack,
        email=email,
        dry_run=args.dry_run,
        sell_only=args.sell_only,
    )


if __name__ == '__main__':
    main()
