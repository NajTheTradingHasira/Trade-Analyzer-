"""
Patch 10: Absolute Strength Alerts
Adds strength-based alert triggers to the alert pipeline.

Alert triggers:
  - NEW LEADER:     abs_strength_score crosses above 8.0 (was below last run)
  - LEADER LOST:    abs_strength_score drops below 6.5 (was leader last run)
  - RS BREAKOUT:    rs_line_direction flips from 'falling'/'flat' to 'rising'
  - RS BREAKDOWN:   rs_line_direction flips from 'rising' to 'falling'
  - RANK SURGE:     abs_rank_20d jumps 30+ percentile points in one scan
  - RANK COLLAPSE:  abs_rank_20d drops 30+ percentile points in one scan
  - DOG ALERT:      abs_strength_score <= 1.5 AND Stage 4 (avoid/short)
  - STAGE+STRENGTH DIVERGENCE: Stage 2 but strength is 'laggard' or 'dog'
                                (false breakout risk)

Dedup: keyed by symbol+alert_type+date in alerts_history.json.
Integrates into the existing alerts.py pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger("as_alerts")

STRENGTH_HISTORY_FILE = Path(__file__).parent.parent / "strength_history.json"


def load_strength_history():
    """Load previous run's strength scores for comparison."""
    if STRENGTH_HISTORY_FILE.exists():
        try:
            with open(STRENGTH_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_strength_history(current):
    """Save current strength snapshot for next run comparison."""
    with open(STRENGTH_HISTORY_FILE, 'w') as f:
        json.dump(current, f, indent=2, default=str)


def detect_strength_alerts(result_df, symbol_col='symbol'):
    """
    Detect absolute strength alert triggers by comparing current scan
    to previous run's strength history.

    Args:
        result_df: stage-enriched DataFrame with abs_strength columns

    Returns:
        list of alert dicts: [{symbol, alert_type, severity, message, score, prev_score, ...}]
    """
    prev = load_strength_history()
    alerts = []
    current_snapshot = {}

    # Get unique tickers with strength data
    if 'abs_strength_score' not in result_df.columns:
        logger.warning("No abs_strength_score column — skipping strength alerts")
        return alerts

    tickers = result_df.drop_duplicates(subset=[symbol_col])

    for _, row in tickers.iterrows():
        sym = row[symbol_col]
        score = row.get('abs_strength_score')
        label = row.get('abs_strength_label', '')
        rank = row.get('abs_rank_20d')
        rs_dir = row.get('rs_line_direction', '')
        rs_spy = row.get('rs_vs_spy_20d')
        mansfield = row.get('mansfield_rs_52w')
        stage = row.get('w_stage')
        stage_label = row.get('w_stage_label', '')
        perf_20d = row.get('abs_perf_20d')

        if score is None or pd.isna(score):
            continue

        # Save snapshot for next run
        current_snapshot[sym] = {
            'score': float(score),
            'label': label,
            'rank': float(rank) if rank is not None and not pd.isna(rank) else None,
            'rs_direction': rs_dir,
            'stage': int(stage) if stage is not None and not pd.isna(stage) else None,
            'timestamp': datetime.utcnow().isoformat(),
        }

        prev_data = prev.get(sym, {})
        prev_score = prev_data.get('score')
        prev_label = prev_data.get('label', '')
        prev_rank = prev_data.get('rank')
        prev_rs_dir = prev_data.get('rs_direction', '')

        base_info = {
            'symbol': sym,
            'score': float(score),
            'prev_score': prev_score,
            'label': label,
            'stage': stage_label,
            'perf_20d': perf_20d,
            'rs_vs_spy': rs_spy,
            'mansfield': mansfield,
            'date': datetime.utcnow().strftime('%Y-%m-%d'),
        }

        # ── NEW LEADER: crosses above 8.0 ──
        if score >= 8.0 and (prev_score is None or prev_score < 8.0):
            alerts.append({
                **base_info,
                'alert_type': 'NEW_LEADER',
                'severity': 'high',
                'message': (
                    f"{sym} is a NEW LEADER (score {score:.1f}, "
                    f"prev {f'{prev_score:.1f}' if prev_score is not None else 'n/a'}). "
                    f"RS vs SPY: {rs_spy:+.1f}%, Mansfield: {mansfield:.1f}. "
                    f"Stage: {stage_label}."
                ),
            })

        # ── LEADER LOST: drops below 6.5 ──
        if prev_label == 'leader' and score < 6.5:
            alerts.append({
                **base_info,
                'alert_type': 'LEADER_LOST',
                'severity': 'high',
                'message': (
                    f"{sym} LOST LEADER STATUS (score {score:.1f}, "
                    f"was {prev_score:.1f}). Now: {label}. "
                    f"Perf 20d: {perf_20d:+.1f}%."
                ),
            })

        # ── RS BREAKOUT: direction flips to rising ──
        if rs_dir == 'rising' and prev_rs_dir in ('falling', 'flat') and prev_rs_dir:
            alerts.append({
                **base_info,
                'alert_type': 'RS_BREAKOUT',
                'severity': 'medium',
                'message': (
                    f"{sym} RS BREAKOUT — RS line turned rising "
                    f"(was {prev_rs_dir}). RS vs SPY: {rs_spy:+.1f}%, "
                    f"Mansfield: {mansfield:.1f}."
                ),
            })

        # ── RS BREAKDOWN: direction flips to falling ──
        if rs_dir == 'falling' and prev_rs_dir == 'rising' and prev_rs_dir:
            alerts.append({
                **base_info,
                'alert_type': 'RS_BREAKDOWN',
                'severity': 'medium',
                'message': (
                    f"{sym} RS BREAKDOWN — RS line turned falling "
                    f"(was rising). RS vs SPY: {rs_spy:+.1f}%, "
                    f"Mansfield: {mansfield:.1f}."
                ),
            })

        # ── RANK SURGE: +30 percentile points ──
        if rank is not None and prev_rank is not None and not pd.isna(rank):
            rank_change = float(rank) - float(prev_rank)
            if rank_change >= 30:
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_SURGE',
                    'severity': 'medium',
                    'message': (
                        f"{sym} RANK SURGE — jumped from {prev_rank:.0f} to "
                        f"{rank:.0f} percentile (+{rank_change:.0f}). "
                        f"Perf 20d: {perf_20d:+.1f}%."
                    ),
                })

            # ── RANK COLLAPSE: -30 percentile points ──
            if rank_change <= -30:
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_COLLAPSE',
                    'severity': 'high',
                    'message': (
                        f"{sym} RANK COLLAPSE — dropped from {prev_rank:.0f} to "
                        f"{rank:.0f} percentile ({rank_change:.0f}). "
                        f"Perf 20d: {perf_20d:+.1f}%."
                    ),
                })

        # ── DOG ALERT: score <= 1.5 AND Stage 4 ──
        if score <= 1.5 and stage == 4:
            alerts.append({
                **base_info,
                'alert_type': 'DOG_ALERT',
                'severity': 'high',
                'message': (
                    f"{sym} is a DOG in Stage 4 (score {score:.1f}). "
                    f"Perf 20d: {perf_20d:+.1f}%, RS vs SPY: {rs_spy:+.1f}%. "
                    f"Avoid longs. Consider bear put spread."
                ),
            })

        # ── STAGE+STRENGTH DIVERGENCE: Stage 2 but weak strength ──
        if stage == 2 and label in ('laggard', 'dog'):
            alerts.append({
                **base_info,
                'alert_type': 'STAGE_STRENGTH_DIVERGENCE',
                'severity': 'medium',
                'message': (
                    f"{sym} DIVERGENCE — Stage 2 (Advancing) but strength "
                    f"is '{label}' (score {score:.1f}). False breakout risk. "
                    f"RS vs SPY: {rs_spy:+.1f}%, Mansfield: {mansfield:.1f}."
                ),
            })

    # Save current snapshot for next run
    save_strength_history(current_snapshot)
    logger.info(f"Strength snapshot saved: {len(current_snapshot)} tickers")

    # Sort by severity then score
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    alerts.sort(key=lambda a: (severity_order.get(a['severity'], 9), -a['score']))

    return alerts


def format_strength_alert_text(alert):
    """Format a single strength alert as plain text."""
    severity_markers = {'high': '!!!', 'medium': '!!', 'low': '!'}
    marker = severity_markers.get(alert['severity'], '')
    return f"[{alert['alert_type']}] {marker} {alert['message']}"


def format_strength_alert_slack(alert):
    """Format a single strength alert as Slack mrkdwn."""
    severity_emoji = {
        'high': ':rotating_light:',
        'medium': ':warning:',
        'low': ':information_source:',
    }
    emoji = severity_emoji.get(alert['severity'], '')
    alert_type = alert['alert_type'].replace('_', ' ')

    return (
        f"{emoji} *{alert_type}* — *{alert['symbol']}*\n"
        f">{alert['message']}"
    )


def print_strength_alerts(alerts):
    """Print strength alerts to console."""
    if not alerts:
        print("\n  No strength alerts triggered.")
        return

    print(f"\n{'='*60}")
    print(f"  ABSOLUTE STRENGTH ALERTS ({len(alerts)})")
    print(f"{'='*60}")

    for alert in alerts:
        severity_markers = {'high': '[HIGH]', 'medium': '[MED] ', 'low': '[LOW] '}
        marker = severity_markers.get(alert['severity'], '')
        print(f"\n  {marker} {alert['alert_type']}")
        print(f"    {alert['message']}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    from patches.add_stage_enriched import stage_enriched_scan

    print("Patch 10: Testing strength alerts...")
    df = pd.read_csv('ohlcv.csv')
    result, regime, stages = stage_enriched_scan(df)

    alerts = detect_strength_alerts(result)
    print_strength_alerts(alerts)

    print(f"\nTotal alerts: {len(alerts)}")
    for a in alerts:
        print(f"  [{a['severity']:6s}] {a['alert_type']:30s} {a['symbol']:6s} score={a['score']:.1f}")
