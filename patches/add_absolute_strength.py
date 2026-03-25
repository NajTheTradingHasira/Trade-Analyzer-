"""
Patch 9: Absolute Strength Overlay
Adds per-ticker absolute strength metrics to the stage-enriched pipeline.

Metrics:
  abs_perf_5d, 10d, 20d, 60d   — raw price performance over N days
  abs_rank_20d                  — percentile rank within universe (0-99)
  rs_vs_spy_20d                 — excess return vs SPY over 20 days
  rs_vs_qqq_20d                 — excess return vs QQQ over 20 days
  mansfield_rs_52w              — Mansfield RS (price/benchmark SMA normalized)
  rs_line_slope_20d             — slope of RS line over 20 days (rising/flat/falling)
  abs_strength_score            — composite 0-10 score blending perf + RS + momentum
  abs_strength_label            — 'leader', 'outperformer', 'inline', 'laggard', 'dog'

Scoring logic:
  - 40% weight: 20-day performance rank within universe
  - 25% weight: RS vs SPY (excess return)
  - 20% weight: Mansfield RS (relative to 52w SMA of RS line)
  - 15% weight: RS line slope (momentum of relative strength)

Integration:
  Called from stage_enriched_scan after stage classification,
  before options overlay. Merges columns into the result DataFrame.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("absolute_strength")


def compute_performance(close_arr, windows=(5, 10, 20, 60)):
    """Compute price performance over multiple windows."""
    n = len(close_arr)
    result = {}
    last = float(close_arr[-1])
    for w in windows:
        if n > w:
            prev = float(close_arr[-(w + 1)])
            result[f'abs_perf_{w}d'] = round(((last / prev) - 1) * 100, 2)
        else:
            result[f'abs_perf_{w}d'] = None
    return result


def compute_rs_vs_benchmark(stock_close, bench_close, window=20):
    """Excess return of stock vs benchmark over N days."""
    if len(stock_close) <= window or len(bench_close) <= window:
        return None
    stock_ret = (float(stock_close[-1]) / float(stock_close[-(window + 1)])) - 1
    bench_ret = (float(bench_close[-1]) / float(bench_close[-(window + 1)])) - 1
    return round((stock_ret - bench_ret) * 100, 2)


def compute_mansfield_rs(stock_close, bench_close, lookback=252):
    """
    Mansfield RS = ((RS / SMA_of_RS) - 1) * 100
    RS = stock_price / benchmark_price
    """
    min_len = min(len(stock_close), len(bench_close))
    if min_len < lookback:
        lookback = max(min_len, 50)
        if lookback < 50:
            return None

    stock = np.array(stock_close[-min_len:], dtype=float)
    bench = np.array(bench_close[-min_len:], dtype=float)

    # Avoid division by zero
    bench = np.where(bench == 0, np.nan, bench)
    rs_line = stock / bench

    valid = rs_line[~np.isnan(rs_line)]
    if len(valid) < lookback:
        return None

    rs_sma = np.mean(valid[-lookback:])
    if rs_sma == 0 or np.isnan(rs_sma):
        return None

    current_rs = valid[-1]
    mansfield = ((current_rs / rs_sma) - 1) * 100
    return round(mansfield, 2)


def compute_rs_line_slope(stock_close, bench_close, window=20):
    """
    Slope of the RS line over N days.
    Positive = strengthening vs benchmark, negative = weakening.
    Returns annualized slope as percentage.
    """
    min_len = min(len(stock_close), len(bench_close))
    if min_len < window + 5:
        return None, None

    stock = np.array(stock_close[-min_len:], dtype=float)
    bench = np.array(bench_close[-min_len:], dtype=float)
    bench = np.where(bench == 0, np.nan, bench)
    rs_line = stock / bench

    recent_rs = rs_line[-window:]
    valid = recent_rs[~np.isnan(recent_rs)]
    if len(valid) < 5:
        return None, None

    # Linear regression slope
    x = np.arange(len(valid))
    slope = np.polyfit(x, valid, 1)[0]

    # Normalize: slope as % of current RS level, annualized
    current_rs = valid[-1]
    if current_rs == 0:
        return None, None
    slope_pct = (slope / current_rs) * 252 * 100  # Annualized

    # Direction label
    if slope_pct > 5:
        direction = 'rising'
    elif slope_pct > -5:
        direction = 'flat'
    else:
        direction = 'falling'

    return round(slope_pct, 2), direction


def score_absolute_strength(perf_rank, rs_vs_spy, mansfield, rs_slope):
    """
    Composite absolute strength score (0-10).

    Weights:
      40% — 20-day performance rank within universe
      25% — RS vs SPY excess return
      20% — Mansfield RS
      15% — RS line slope
    """
    score = 0.0

    # Performance rank (0-99 → 0-4 points)
    if perf_rank is not None:
        score += (perf_rank / 99) * 4.0

    # RS vs SPY (-20% to +20% → 0-2.5 points)
    if rs_vs_spy is not None:
        rs_clamped = max(-20, min(20, rs_vs_spy))
        score += ((rs_clamped + 20) / 40) * 2.5

    # Mansfield RS (-10 to +10 → 0-2 points)
    if mansfield is not None:
        m_clamped = max(-10, min(10, mansfield))
        score += ((m_clamped + 10) / 20) * 2.0

    # RS slope (-50 to +50 → 0-1.5 points)
    if rs_slope is not None:
        s_clamped = max(-50, min(50, rs_slope))
        score += ((s_clamped + 50) / 100) * 1.5

    return round(min(score, 10.0), 1)


def label_strength(score):
    """Map score to human-readable label."""
    if score >= 8.0:
        return 'leader'
    elif score >= 6.5:
        return 'outperformer'
    elif score >= 4.0:
        return 'inline'
    elif score >= 2.0:
        return 'laggard'
    else:
        return 'dog'


def compute_absolute_strength_universe(ohlcv_df, symbol_col='symbol', date_col='date',
                                        close_col='close', benchmark_symbols=('SPY', 'QQQ')):
    """
    Compute absolute strength metrics for all tickers in the universe.

    Args:
        ohlcv_df: OHLCV DataFrame (multi-symbol)
        benchmark_symbols: tuple of benchmark tickers for RS computation

    Returns:
        dict: {symbol: {abs_perf_5d, abs_perf_10d, ..., abs_strength_score, abs_strength_label}}
    """
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df[date_col] = pd.to_datetime(ohlcv_df[date_col])
    ohlcv_df = ohlcv_df.sort_values([symbol_col, date_col])

    # Build close arrays
    close_arrays = {}
    for sym, g in ohlcv_df.groupby(symbol_col):
        close_arrays[sym] = g[close_col].values

    # Get benchmark arrays
    bench_data = {}
    for b_sym in benchmark_symbols:
        if b_sym in close_arrays:
            bench_data[b_sym] = close_arrays[b_sym]

    spy_close = bench_data.get('SPY')
    qqq_close = bench_data.get('QQQ')

    # Compute per-ticker metrics
    all_results = {}
    perf_20d_values = {}  # For ranking

    symbols = [s for s in close_arrays if s not in benchmark_symbols]

    for sym in symbols:
        close = close_arrays[sym]
        if len(close) < 30:
            all_results[sym] = _empty_result()
            continue

        # Performance
        perf = compute_performance(close)

        # RS vs benchmarks
        rs_spy = compute_rs_vs_benchmark(close, spy_close) if spy_close is not None else None
        rs_qqq = compute_rs_vs_benchmark(close, qqq_close) if qqq_close is not None else None

        # Mansfield RS (vs SPY)
        mansfield = compute_mansfield_rs(close, spy_close) if spy_close is not None else None

        # RS line slope (vs SPY)
        rs_slope, rs_direction = compute_rs_line_slope(close, spy_close) if spy_close is not None else (None, None)

        result = {
            **perf,
            'rs_vs_spy_20d': rs_spy,
            'rs_vs_qqq_20d': rs_qqq,
            'mansfield_rs_52w': mansfield,
            'rs_line_slope_20d': rs_slope,
            'rs_line_direction': rs_direction,
        }

        all_results[sym] = result
        if perf.get('abs_perf_20d') is not None:
            perf_20d_values[sym] = perf['abs_perf_20d']

    # Compute universe rank for 20-day performance
    if perf_20d_values:
        sorted_syms = sorted(perf_20d_values.items(), key=lambda x: x[1])
        n = len(sorted_syms)
        for rank_idx, (sym, _) in enumerate(sorted_syms):
            percentile = round((rank_idx / max(n - 1, 1)) * 99, 0)
            all_results[sym]['abs_rank_20d'] = percentile

    # Compute composite score
    for sym, data in all_results.items():
        if data.get('abs_perf_5d') is None:
            data['abs_strength_score'] = None
            data['abs_strength_label'] = None
            continue

        score = score_absolute_strength(
            data.get('abs_rank_20d'),
            data.get('rs_vs_spy_20d'),
            data.get('mansfield_rs_52w'),
            data.get('rs_line_slope_20d'),
        )
        data['abs_strength_score'] = score
        data['abs_strength_label'] = label_strength(score)

    return all_results


def _empty_result():
    return {
        'abs_perf_5d': None, 'abs_perf_10d': None,
        'abs_perf_20d': None, 'abs_perf_60d': None,
        'abs_rank_20d': None,
        'rs_vs_spy_20d': None, 'rs_vs_qqq_20d': None,
        'mansfield_rs_52w': None,
        'rs_line_slope_20d': None, 'rs_line_direction': None,
        'abs_strength_score': None, 'abs_strength_label': None,
    }


def apply_absolute_strength(df, ohlcv_df, symbol_col='symbol', date_col='date',
                             close_col='close'):
    """
    Apply absolute strength overlay to an enriched scan DataFrame.
    Merges per-ticker strength metrics into every row.
    """
    strength = compute_absolute_strength_universe(ohlcv_df, symbol_col, date_col, close_col)

    # Build merge DataFrame
    rows = []
    for sym, data in strength.items():
        row = {'symbol': sym}
        row.update(data)
        rows.append(row)

    if not rows:
        return df

    strength_df = pd.DataFrame(rows)
    merge_cols = [c for c in strength_df.columns if c != symbol_col]

    # Drop any existing columns to avoid conflicts
    for col in merge_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(strength_df, on=symbol_col, how='left')

    # Print summary
    scored = strength_df.dropna(subset=['abs_strength_score'])
    if not scored.empty:
        label_dist = scored['abs_strength_label'].value_counts()
        logger.info(f"Absolute strength applied to {len(scored)} tickers")

        print(f"\n  Absolute Strength distribution:")
        for label, count in label_dist.items():
            bar = '#' * (count * 2)
            print(f"    {label:15s}: {count:3d}  {bar}")

        # Top 5 and bottom 5
        top5 = scored.nlargest(5, 'abs_strength_score')
        bot5 = scored.nsmallest(5, 'abs_strength_score')

        print(f"\n  Top 5 (leaders):")
        for _, r in top5.iterrows():
            print(f"    {r['symbol']:6s}  score={r['abs_strength_score']:.1f}  "
                  f"perf_20d={r.get('abs_perf_20d', 'n/a')}%  "
                  f"RS_spy={r.get('rs_vs_spy_20d', 'n/a')}%  "
                  f"mansfield={r.get('mansfield_rs_52w', 'n/a')}")

        print(f"\n  Bottom 5 (laggards):")
        for _, r in bot5.iterrows():
            print(f"    {r['symbol']:6s}  score={r['abs_strength_score']:.1f}  "
                  f"perf_20d={r.get('abs_perf_20d', 'n/a')}%  "
                  f"RS_spy={r.get('rs_vs_spy_20d', 'n/a')}%  "
                  f"mansfield={r.get('mansfield_rs_52w', 'n/a')}")

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 9: Testing absolute strength...")
    df = pd.read_csv('ohlcv.csv')
    df['date'] = pd.to_datetime(df['date'])

    results = compute_absolute_strength_universe(df)

    # Print leaderboard
    scored = [(sym, d) for sym, d in results.items() if d.get('abs_strength_score') is not None]
    scored.sort(key=lambda x: x[1]['abs_strength_score'], reverse=True)

    print(f"\n{'='*80}")
    print(f"  ABSOLUTE STRENGTH LEADERBOARD ({len(scored)} tickers)")
    print(f"{'='*80}")
    print(f"  {'Ticker':8s} {'Score':>6s} {'Label':>14s} {'Perf20d':>8s} {'RS_SPY':>8s} {'Mansfield':>10s} {'RS_Slope':>9s} {'Dir':>8s}")
    print(f"  {'-'*74}")
    for sym, d in scored:
        print(f"  {sym:8s} {d['abs_strength_score']:6.1f} {d.get('abs_strength_label',''):>14s} "
              f"{d.get('abs_perf_20d', 0):7.1f}% {d.get('rs_vs_spy_20d', 0):7.1f}% "
              f"{d.get('mansfield_rs_52w', 0):9.1f} {d.get('rs_line_slope_20d', 0):8.1f} "
              f"{d.get('rs_line_direction', ''):>8s}")
