"""
Patch 13: Absolute Strength Monitor + Rating Alerts
Adds multi-timeframe momentum scoring with RS line tracking and tiered alerts.

absolute_strength_monitor():
  Scores 7 attributes (0 or 1 each) → composite 0-7:
    1. Price > 20d SMA
    2. Price > 50d SMA
    3. 20d SMA > 50d SMA (golden cross)
    4. RS line at/near 20-day high (within 2%)
    5. New 52-week high or within 3% of 52w high
    6. New 20-day high
    7. Volume confirmation (5d avg vol > 50d avg vol * 1.2)

  Additional metrics:
    as_percentile_rank   — percentile within universe (0-100)
    rs_line_at_20d_high  — bool, RS line near its own 20d high
    flagged              — bool, score >= 4 AND rs_line_at_20d_high

as_rating_alerts():
  Tiered alert levels based on percentile:
    TIER_1 (80th pctile): "Strong" — worth watching
    TIER_2 (90th pctile): "Very Strong" — active candidate
    TIER_3 (95th pctile): "Elite" — top of universe, highest conviction

  Auto-applied when running --mode absolute_strength.

CLI: --mode absolute_strength
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("as_monitor")

TIER_THRESHOLDS = {
    'TIER_3_ELITE': 95,
    'TIER_2_VERY_STRONG': 90,
    'TIER_1_STRONG': 80,
}


# ══════════════════════════════════════════════════════════════
# 1. SCORING ENGINE
# ══════════════════════════════════════════════════════════════

def _score_single_ticker(close, volume, bench_close=None):
    """
    Score a single ticker on 7 attributes (0 or 1 each).
    Returns dict with individual flags, composite score, and RS metrics.
    """
    n = len(close)
    result = {
        'above_20sma': False,
        'above_50sma': False,
        'golden_cross_20_50': False,
        'rs_at_20d_high': False,
        'near_52w_high': False,
        'new_20d_high': False,
        'volume_confirmation': False,
        'as_composite_score': 0,
        'rs_line_value': None,
        'rs_line_20d_high': None,
        'rs_line_pct_from_20d_high': None,
    }

    if n < 50:
        return result

    last = float(close[-1])

    # ── 1. Price > 20d SMA ──
    sma_20 = float(np.mean(close[-20:]))
    result['above_20sma'] = last > sma_20

    # ── 2. Price > 50d SMA ──
    sma_50 = float(np.mean(close[-50:]))
    result['above_50sma'] = last > sma_50

    # ── 3. 20d SMA > 50d SMA (golden cross) ──
    result['golden_cross_20_50'] = sma_20 > sma_50

    # ── 4. RS line at/near 20-day high ──
    if bench_close is not None and len(bench_close) >= n:
        min_len = min(n, len(bench_close))
        stock_tail = close[-min_len:]
        bench_tail = bench_close[-min_len:]

        # Avoid zero division
        bench_safe = np.where(bench_tail == 0, np.nan, bench_tail)
        rs_line = stock_tail / bench_safe
        rs_valid = rs_line[~np.isnan(rs_line)]

        if len(rs_valid) >= 20:
            rs_current = float(rs_valid[-1])
            rs_20d = rs_valid[-20:]
            rs_20d_high = float(np.max(rs_20d))

            result['rs_line_value'] = round(rs_current, 6)
            result['rs_line_20d_high'] = round(rs_20d_high, 6)

            if rs_20d_high > 0:
                pct_from_high = ((rs_current / rs_20d_high) - 1) * 100
                result['rs_line_pct_from_20d_high'] = round(pct_from_high, 2)
                result['rs_at_20d_high'] = pct_from_high >= -2.0  # Within 2%

    # ── 5. Near 52-week high (within 3%) ──
    lookback_52w = min(252, n)
    high_52w = float(np.max(close[-lookback_52w:]))
    if high_52w > 0:
        pct_from_52w = ((last / high_52w) - 1) * 100
        result['near_52w_high'] = pct_from_52w >= -3.0
        result['pct_from_52w_high'] = round(pct_from_52w, 2)

    # ── 6. New 20-day high ──
    high_20d = float(np.max(close[-20:]))
    result['new_20d_high'] = last >= high_20d * 0.998  # Within 0.2%

    # ── 7. Volume confirmation ──
    if len(volume) >= 50:
        vol_5d_avg = float(np.mean(volume[-5:]))
        vol_50d_avg = float(np.mean(volume[-50:]))
        if vol_50d_avg > 0:
            result['volume_confirmation'] = vol_5d_avg >= vol_50d_avg * 1.2
            result['vol_ratio_5d_50d'] = round(vol_5d_avg / vol_50d_avg, 2)

    # ── Composite score (0-7) ──
    score = sum([
        result['above_20sma'],
        result['above_50sma'],
        result['golden_cross_20_50'],
        result['rs_at_20d_high'],
        result['near_52w_high'],
        result['new_20d_high'],
        result['volume_confirmation'],
    ])
    result['as_composite_score'] = score

    # ── Flag: score >= 4 AND RS at 20d high ──
    result['as_flagged'] = score >= 4 and result['rs_at_20d_high']

    return result


def absolute_strength_monitor(ohlcv_df, symbol_col='symbol', date_col='date',
                               close_col='close', volume_col='volume',
                               benchmark_symbol='SPY'):
    """
    Score every ticker in the universe on 7 momentum attributes.
    Returns dict: {symbol: {score, flags, percentile_rank, flagged, ...}}
    """
    df = ohlcv_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])

    # Build arrays
    close_arrays = {}
    volume_arrays = {}
    for sym, g in df.groupby(symbol_col):
        close_arrays[sym] = g[close_col].values.astype(float)
        volume_arrays[sym] = g[volume_col].values.astype(float)

    bench_close = close_arrays.get(benchmark_symbol)

    # Score each ticker
    results = {}
    scores = {}

    symbols = [s for s in close_arrays if s != benchmark_symbol]

    for sym in symbols:
        close = close_arrays[sym]
        volume = volume_arrays.get(sym, np.array([]))

        data = _score_single_ticker(close, volume, bench_close)
        results[sym] = data
        if data['as_composite_score'] is not None:
            scores[sym] = data['as_composite_score']

    # ── Percentile rank within universe ──
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        n = len(sorted_scores)
        for rank_idx, (sym, _) in enumerate(sorted_scores):
            pctile = round((rank_idx / max(n - 1, 1)) * 100, 1)
            results[sym]['as_percentile_rank'] = pctile

    return results


# ══════════════════════════════════════════════════════════════
# 2. TIERED RATING ALERTS
# ══════════════════════════════════════════════════════════════

def as_rating_alerts(monitor_results):
    """
    Generate tiered alerts based on percentile rank:
      TIER_3 (>=95th): "Elite" — top of universe
      TIER_2 (>=90th): "Very Strong" — active candidate
      TIER_1 (>=80th): "Strong" — worth watching

    Also flags tickers with score >= 4 AND RS at 20d high.

    Returns list of alert dicts.
    """
    alerts = []
    ts = datetime.utcnow().strftime('%Y-%m-%d')

    for sym, data in monitor_results.items():
        pctile = data.get('as_percentile_rank')
        score = data.get('as_composite_score', 0)
        flagged = data.get('as_flagged', False)

        if pctile is None:
            continue

        base = {
            'symbol': sym,
            'as_composite_score': score,
            'as_percentile_rank': pctile,
            'above_20sma': data.get('above_20sma'),
            'above_50sma': data.get('above_50sma'),
            'golden_cross': data.get('golden_cross_20_50'),
            'rs_at_20d_high': data.get('rs_at_20d_high'),
            'near_52w_high': data.get('near_52w_high'),
            'new_20d_high': data.get('new_20d_high'),
            'volume_confirmation': data.get('volume_confirmation'),
            'rs_pct_from_20d_high': data.get('rs_line_pct_from_20d_high'),
            'pct_from_52w_high': data.get('pct_from_52w_high'),
            'flagged': flagged,
            'date': ts,
        }

        # ── Tier classification ──
        tier = None
        tier_label = None
        severity = 'low'

        if pctile >= TIER_THRESHOLDS['TIER_3_ELITE']:
            tier = 'TIER_3'
            tier_label = 'ELITE'
            severity = 'high'
        elif pctile >= TIER_THRESHOLDS['TIER_2_VERY_STRONG']:
            tier = 'TIER_2'
            tier_label = 'VERY STRONG'
            severity = 'high'
        elif pctile >= TIER_THRESHOLDS['TIER_1_STRONG']:
            tier = 'TIER_1'
            tier_label = 'STRONG'
            severity = 'medium'

        if tier is None:
            continue

        # Build attribute string
        attrs = []
        if data.get('above_20sma'): attrs.append('>20SMA')
        if data.get('above_50sma'): attrs.append('>50SMA')
        if data.get('golden_cross_20_50'): attrs.append('GoldenX')
        if data.get('rs_at_20d_high'): attrs.append('RS@20dHi')
        if data.get('near_52w_high'): attrs.append('Nr52wHi')
        if data.get('new_20d_high'): attrs.append('New20dHi')
        if data.get('volume_confirmation'): attrs.append('VolConf')
        attr_str = ' | '.join(attrs)

        message = (
            f"{sym} [{tier_label}] Score {score}/7, {pctile:.0f}th percentile. "
            f"Attributes: {attr_str}."
        )

        if flagged:
            message += " ** FLAGGED: score>=4 + RS at 20d high **"

        alerts.append({
            **base,
            'alert_type': f'AS_RATING_{tier}',
            'tier': tier,
            'tier_label': tier_label,
            'severity': severity,
            'message': message,
        })

    # Sort by percentile descending
    alerts.sort(key=lambda a: -a['as_percentile_rank'])

    return alerts


# ══════════════════════════════════════════════════════════════
# 3. DISPLAY + INTEGRATION
# ══════════════════════════════════════════════════════════════

def print_monitor_table(results):
    """Print the full monitor leaderboard."""
    scored = [(sym, d) for sym, d in results.items() if d.get('as_composite_score') is not None]
    scored.sort(key=lambda x: (-x[1]['as_composite_score'], -(x[1].get('as_percentile_rank') or 0)))

    print(f"\n{'='*100}")
    print(f"  ABSOLUTE STRENGTH MONITOR — {len(scored)} tickers")
    print(f"{'='*100}")
    print(f"  {'Ticker':7s} {'Score':>5s} {'Pctile':>7s} {'Flag':>5s} "
          f"{'20SMA':>6s} {'50SMA':>6s} {'GoldX':>6s} {'RS20H':>6s} "
          f"{'52wHi':>6s} {'20dHi':>6s} {'VolCf':>6s} "
          f"{'RS%20dH':>8s} {'%52wH':>7s}")
    print(f"  {'-'*95}")

    for sym, d in scored:
        flag = '>>>' if d.get('as_flagged') else ''
        print(f"  {sym:7s} {d['as_composite_score']:5d} "
              f"{d.get('as_percentile_rank', 0):6.0f}% "
              f"{flag:>5s} "
              f"{'Y' if d.get('above_20sma') else '.':>6s} "
              f"{'Y' if d.get('above_50sma') else '.':>6s} "
              f"{'Y' if d.get('golden_cross_20_50') else '.':>6s} "
              f"{'Y' if d.get('rs_at_20d_high') else '.':>6s} "
              f"{'Y' if d.get('near_52w_high') else '.':>6s} "
              f"{'Y' if d.get('new_20d_high') else '.':>6s} "
              f"{'Y' if d.get('volume_confirmation') else '.':>6s} "
              f"{(d.get('rs_line_pct_from_20d_high') or 0):>+7.1f}% "
              f"{(d.get('pct_from_52w_high') or 0):>+6.1f}%")

    print(f"{'='*100}")

    # Score distribution
    score_counts = {}
    for _, d in scored:
        s = d['as_composite_score']
        score_counts[s] = score_counts.get(s, 0) + 1

    print(f"\n  Score distribution:")
    for s in sorted(score_counts.keys(), reverse=True):
        bar = '#' * (score_counts[s] * 2)
        print(f"    {s}/7: {score_counts[s]:3d}  {bar}")

    # Flagged tickers
    flagged = [(sym, d) for sym, d in scored if d.get('as_flagged')]
    if flagged:
        print(f"\n  FLAGGED (score>=4 + RS at 20d high): {len(flagged)}")
        for sym, d in flagged:
            print(f"    {sym:7s}  score={d['as_composite_score']}/7  "
                  f"pctile={d.get('as_percentile_rank', 0):.0f}%  "
                  f"RS%={d.get('rs_line_pct_from_20d_high', 0):+.1f}%")


def print_rating_alerts(alerts):
    """Print tiered rating alerts."""
    if not alerts:
        print("\n  No tickers above 80th percentile threshold.")
        return

    print(f"\n{'='*80}")
    print(f"  AS RATING ALERTS ({len(alerts)})")
    print(f"{'='*80}")

    for tier in ['TIER_3', 'TIER_2', 'TIER_1']:
        tier_alerts = [a for a in alerts if a['tier'] == tier]
        if not tier_alerts:
            continue

        tier_labels = {'TIER_3': 'ELITE (>=95th)', 'TIER_2': 'VERY STRONG (>=90th)', 'TIER_1': 'STRONG (>=80th)'}
        print(f"\n  --- {tier_labels[tier]} --- ({len(tier_alerts)} tickers)")

        for a in tier_alerts:
            flag_marker = ' ***' if a['flagged'] else ''
            print(f"    {a['symbol']:7s}  score={a['as_composite_score']}/7  "
                  f"pctile={a['as_percentile_rank']:.0f}%{flag_marker}")
            # Show which attributes are active
            attrs = []
            if a['above_20sma']: attrs.append('>20SMA')
            if a['above_50sma']: attrs.append('>50SMA')
            if a['golden_cross']: attrs.append('GoldenX')
            if a['rs_at_20d_high']: attrs.append('RS@20dHi')
            if a['near_52w_high']: attrs.append('Nr52wHi')
            if a['new_20d_high']: attrs.append('New20dHi')
            if a['volume_confirmation']: attrs.append('VolConf')
            print(f"             [{' | '.join(attrs)}]")

    print(f"\n{'='*80}")


def run_absolute_strength_mode(ohlcv_df, symbol_col='symbol', date_col='date',
                                close_col='close', volume_col='volume',
                                benchmark_symbol='SPY'):
    """
    Full absolute strength monitor pipeline:
      1. Score all tickers (7 attributes)
      2. Rank within universe
      3. Generate tiered rating alerts
      4. Print leaderboard + alerts

    Returns (monitor_results, rating_alerts)
    """
    print("\n" + "=" * 60)
    print("  ABSOLUTE STRENGTH MONITOR")
    print("=" * 60)

    # 1. Score universe
    results = absolute_strength_monitor(
        ohlcv_df, symbol_col, date_col, close_col, volume_col, benchmark_symbol
    )

    # 2. Print leaderboard
    print_monitor_table(results)

    # 3. Generate + print rating alerts
    alerts = as_rating_alerts(results)
    print_rating_alerts(alerts)

    # 4. Summary
    total = len(results)
    flagged = sum(1 for d in results.values() if d.get('as_flagged'))
    above_80 = sum(1 for d in results.values()
                   if (d.get('as_percentile_rank') or 0) >= 80)

    print(f"\n  Monitor summary:")
    print(f"    Total tickers:    {total}")
    print(f"    Flagged (S>=4+RS): {flagged}")
    print(f"    Above 80th pctile: {above_80}")
    print(f"    Rating alerts:    {len(alerts)}")

    return results, alerts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 13: Testing absolute strength monitor...")
    df = pd.read_csv('ohlcv.csv')

    results, alerts = run_absolute_strength_mode(df)

    # Save results as CSV
    rows = []
    for sym, d in results.items():
        rows.append({'symbol': sym, **d})
    pd.DataFrame(rows).sort_values('as_composite_score', ascending=False).to_csv(
        'as_monitor_results.csv', index=False
    )
    print("\nSaved to as_monitor_results.csv")
