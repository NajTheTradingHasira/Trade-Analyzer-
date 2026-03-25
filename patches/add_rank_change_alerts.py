"""
Patch 11: Rank Change Alerts
Tracks week-over-week and day-over-day rank changes with rolling
history for trend detection.

Alert triggers:
  - RANK MOMENTUM UP:    3+ consecutive scans of rank improvement (gaining ground)
  - RANK MOMENTUM DOWN:  3+ consecutive scans of rank deterioration (losing ground)
  - RANK INFLECTION UP:  rank was falling for 3+ scans, now rising (potential bottom)
  - RANK INFLECTION DOWN: rank was rising for 3+ scans, now falling (potential top)
  - PERCENTILE EXTREME:  rank hits top 5 (>=95) or bottom 5 (<=5) — crowded territory
  - LABEL PROMOTION:     strength label upgraded (e.g., laggard -> inline -> outperformer)
  - LABEL DEMOTION:      strength label downgraded
  - FASTEST RISER:       largest positive rank change in universe this scan
  - FASTEST FALLER:      largest negative rank change in universe this scan

Stores rolling rank history (last 20 scans) per ticker in rank_history.json.
Integrates into the alert pipeline alongside add_as_alerts.py.
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

logger = logging.getLogger("rank_change_alerts")

RANK_HISTORY_FILE = Path(__file__).parent.parent / "rank_history.json"
MAX_HISTORY_LENGTH = 20  # Keep last 20 scans per ticker

LABEL_ORDER = ['dog', 'laggard', 'inline', 'outperformer', 'leader']


def load_rank_history():
    """Load rolling rank history. Structure: {symbol: [{rank, label, score, ts}, ...]}"""
    if RANK_HISTORY_FILE.exists():
        try:
            with open(RANK_HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_rank_history(history):
    with open(RANK_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def _rank_trend(ranks, n=3):
    """
    Determine rank trend from last N data points.
    Returns: 'rising', 'falling', 'flat', or None if insufficient data.
    """
    if len(ranks) < n:
        return None
    recent = ranks[-n:]
    diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    if all(d > 0 for d in diffs):
        return 'rising'
    if all(d < 0 for d in diffs):
        return 'falling'
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff > 2:
        return 'rising'
    if avg_diff < -2:
        return 'falling'
    return 'flat'


def _label_index(label):
    """Convert label to ordinal for comparison."""
    try:
        return LABEL_ORDER.index(label)
    except ValueError:
        return 2  # default to 'inline'


def detect_rank_change_alerts(result_df, symbol_col='symbol'):
    """
    Detect rank change alerts by comparing current scan to rolling history.

    Args:
        result_df: stage-enriched DataFrame with abs_strength columns

    Returns:
        list of alert dicts
    """
    history = load_rank_history()
    alerts = []

    if 'abs_rank_20d' not in result_df.columns:
        logger.warning("No abs_rank_20d column — skipping rank change alerts")
        return alerts

    tickers = result_df.drop_duplicates(subset=[symbol_col])
    ts = datetime.utcnow().isoformat()

    # Collect current scan data for universe-wide comparisons
    current_changes = {}

    for _, row in tickers.iterrows():
        sym = row[symbol_col]
        rank = row.get('abs_rank_20d')
        label = row.get('abs_strength_label', '')
        score = row.get('abs_strength_score')
        stage_label = row.get('w_stage_label', '')
        perf_20d = row.get('abs_perf_20d')
        rs_spy = row.get('rs_vs_spy_20d')

        if rank is None or pd.isna(rank):
            continue

        rank = float(rank)
        score = float(score) if score is not None and not pd.isna(score) else 0

        # Get history for this ticker
        sym_history = history.get(sym, [])
        prev_ranks = [h['rank'] for h in sym_history if h.get('rank') is not None]
        prev_labels = [h['label'] for h in sym_history if h.get('label')]

        # Compute rank change vs last scan
        rank_change = None
        if prev_ranks:
            rank_change = rank - prev_ranks[-1]
            current_changes[sym] = rank_change

        prev_label = prev_labels[-1] if prev_labels else None

        base_info = {
            'symbol': sym,
            'score': score,
            'rank': rank,
            'prev_rank': prev_ranks[-1] if prev_ranks else None,
            'rank_change': rank_change,
            'label': label,
            'prev_label': prev_label,
            'stage': stage_label,
            'perf_20d': perf_20d,
            'rs_vs_spy': rs_spy,
            'date': datetime.utcnow().strftime('%Y-%m-%d'),
        }

        # Need at least 3 historical points for trend alerts
        if len(prev_ranks) >= 3:
            # Current trend (including this scan)
            all_ranks = prev_ranks + [rank]
            current_trend = _rank_trend(all_ranks[-4:])
            prior_trend = _rank_trend(prev_ranks[-3:])

            # ── RANK MOMENTUM UP: 3+ consecutive improvements ──
            if current_trend == 'rising' and len(all_ranks) >= 4:
                total_gain = rank - all_ranks[-4]
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_MOMENTUM_UP',
                    'severity': 'medium',
                    'message': (
                        f"{sym} RANK MOMENTUM UP — rising for 3+ scans "
                        f"(rank {all_ranks[-4]:.0f} -> {rank:.0f}, +{total_gain:.0f} pts). "
                        f"Score: {score:.1f}, RS vs SPY: {rs_spy:+.1f}%."
                    ),
                })

            # ── RANK MOMENTUM DOWN: 3+ consecutive deteriorations ──
            if current_trend == 'falling' and len(all_ranks) >= 4:
                total_loss = rank - all_ranks[-4]
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_MOMENTUM_DOWN',
                    'severity': 'medium',
                    'message': (
                        f"{sym} RANK MOMENTUM DOWN — falling for 3+ scans "
                        f"(rank {all_ranks[-4]:.0f} -> {rank:.0f}, {total_loss:.0f} pts). "
                        f"Score: {score:.1f}, Perf 20d: {perf_20d:+.1f}%."
                    ),
                })

            # ── RANK INFLECTION UP: was falling, now rising ──
            if prior_trend == 'falling' and rank_change and rank_change > 3:
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_INFLECTION_UP',
                    'severity': 'high',
                    'message': (
                        f"{sym} RANK INFLECTION UP — was falling, now turning up "
                        f"(rank {prev_ranks[-1]:.0f} -> {rank:.0f}, +{rank_change:.0f}). "
                        f"Potential bottom in relative strength."
                    ),
                })

            # ── RANK INFLECTION DOWN: was rising, now falling ──
            if prior_trend == 'rising' and rank_change and rank_change < -3:
                alerts.append({
                    **base_info,
                    'alert_type': 'RANK_INFLECTION_DOWN',
                    'severity': 'high',
                    'message': (
                        f"{sym} RANK INFLECTION DOWN — was rising, now turning down "
                        f"(rank {prev_ranks[-1]:.0f} -> {rank:.0f}, {rank_change:.0f}). "
                        f"Potential top in relative strength."
                    ),
                })

        # ── PERCENTILE EXTREME: top 5 or bottom 5 ──
        if rank >= 95:
            alerts.append({
                **base_info,
                'alert_type': 'PERCENTILE_EXTREME_HIGH',
                'severity': 'low',
                'message': (
                    f"{sym} at {rank:.0f}th percentile — top of universe. "
                    f"Score: {score:.1f}. Watch for mean reversion risk."
                ),
            })
        elif rank <= 5:
            alerts.append({
                **base_info,
                'alert_type': 'PERCENTILE_EXTREME_LOW',
                'severity': 'low',
                'message': (
                    f"{sym} at {rank:.0f}th percentile — bottom of universe. "
                    f"Score: {score:.1f}. Deep relative weakness."
                ),
            })

        # ── LABEL PROMOTION ──
        if prev_label and label:
            curr_idx = _label_index(label)
            prev_idx = _label_index(prev_label)
            if curr_idx > prev_idx:
                alerts.append({
                    **base_info,
                    'alert_type': 'LABEL_PROMOTION',
                    'severity': 'medium' if curr_idx >= 3 else 'low',
                    'message': (
                        f"{sym} PROMOTED: {prev_label} -> {label} "
                        f"(score {score:.1f}). Improving relative strength."
                    ),
                })

            # ── LABEL DEMOTION ──
            elif curr_idx < prev_idx:
                alerts.append({
                    **base_info,
                    'alert_type': 'LABEL_DEMOTION',
                    'severity': 'medium' if prev_idx >= 3 else 'low',
                    'message': (
                        f"{sym} DEMOTED: {prev_label} -> {label} "
                        f"(score {score:.1f}). Deteriorating relative strength."
                    ),
                })

        # Update history for this ticker
        new_entry = {'rank': rank, 'label': label, 'score': score, 'ts': ts}
        sym_history.append(new_entry)
        if len(sym_history) > MAX_HISTORY_LENGTH:
            sym_history = sym_history[-MAX_HISTORY_LENGTH:]
        history[sym] = sym_history

    # ── FASTEST RISER / FALLER (universe-wide) ──
    if current_changes:
        fastest_riser_sym = max(current_changes, key=current_changes.get)
        fastest_faller_sym = min(current_changes, key=current_changes.get)

        riser_change = current_changes[fastest_riser_sym]
        faller_change = current_changes[fastest_faller_sym]

        if riser_change > 5:
            riser_row = tickers[tickers[symbol_col] == fastest_riser_sym].iloc[0]
            alerts.append({
                'symbol': fastest_riser_sym,
                'score': float(riser_row.get('abs_strength_score', 0)),
                'rank': float(riser_row.get('abs_rank_20d', 0)),
                'rank_change': riser_change,
                'label': riser_row.get('abs_strength_label', ''),
                'stage': riser_row.get('w_stage_label', ''),
                'perf_20d': riser_row.get('abs_perf_20d'),
                'rs_vs_spy': riser_row.get('rs_vs_spy_20d'),
                'date': datetime.utcnow().strftime('%Y-%m-%d'),
                'alert_type': 'FASTEST_RISER',
                'severity': 'medium',
                'message': (
                    f"{fastest_riser_sym} is the FASTEST RISER in the universe "
                    f"(rank +{riser_change:.0f} pts this scan). "
                    f"Score: {float(riser_row.get('abs_strength_score', 0)):.1f}."
                ),
            })

        if faller_change < -5:
            faller_row = tickers[tickers[symbol_col] == fastest_faller_sym].iloc[0]
            alerts.append({
                'symbol': fastest_faller_sym,
                'score': float(faller_row.get('abs_strength_score', 0)),
                'rank': float(faller_row.get('abs_rank_20d', 0)),
                'rank_change': faller_change,
                'label': faller_row.get('abs_strength_label', ''),
                'stage': faller_row.get('w_stage_label', ''),
                'perf_20d': faller_row.get('abs_perf_20d'),
                'rs_vs_spy': faller_row.get('rs_vs_spy_20d'),
                'date': datetime.utcnow().strftime('%Y-%m-%d'),
                'alert_type': 'FASTEST_FALLER',
                'severity': 'medium',
                'message': (
                    f"{fastest_faller_sym} is the FASTEST FALLER in the universe "
                    f"(rank {faller_change:.0f} pts this scan). "
                    f"Score: {float(faller_row.get('abs_strength_score', 0)):.1f}."
                ),
            })

    # Save updated history
    save_rank_history(history)
    logger.info(f"Rank history updated: {len(history)} tickers, max {MAX_HISTORY_LENGTH} scans each")

    # Sort by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    alerts.sort(key=lambda a: (severity_order.get(a['severity'], 9), -(a.get('score', 0))))

    return alerts


def format_rank_alert_slack(alert):
    """Format a rank change alert for Slack."""
    severity_emoji = {
        'high': ':chart_with_upwards_trend:' if 'UP' in alert['alert_type'] else ':chart_with_downwards_trend:',
        'medium': ':arrow_up:' if 'UP' in alert['alert_type'] or 'RISER' in alert['alert_type'] else ':arrow_down:',
        'low': ':small_blue_diamond:',
    }
    emoji = severity_emoji.get(alert['severity'], '')
    alert_type = alert['alert_type'].replace('_', ' ')

    return (
        f"{emoji} *{alert_type}* — *{alert['symbol']}*\n"
        f">{alert['message']}"
    )


def print_rank_change_alerts(alerts):
    """Print rank change alerts to console."""
    if not alerts:
        print("\n  No rank change alerts triggered.")
        return

    print(f"\n{'='*60}")
    print(f"  RANK CHANGE ALERTS ({len(alerts)})")
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

    print("Patch 11: Testing rank change alerts...")
    df = pd.read_csv('ohlcv.csv')
    result, regime, stages = stage_enriched_scan(df)

    # Run twice to simulate consecutive scans
    print("\n--- First scan (baseline) ---")
    alerts1 = detect_rank_change_alerts(result)
    print_rank_change_alerts(alerts1)

    print("\n--- Second scan (should detect changes if any) ---")
    alerts2 = detect_rank_change_alerts(result)
    print_rank_change_alerts(alerts2)

    print(f"\nRank history file: {RANK_HISTORY_FILE}")
