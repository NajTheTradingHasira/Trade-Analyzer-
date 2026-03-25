"""
Patch 1: Gap Fill Risk Analysis
Adds: gap_fill_risk, filled_same_day, fill_distance_pct

Measures how likely a gap is to fill and how far price traveled
toward filling the gap on the gap day itself.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from breakaway_gap_scan import breakaway_gap_scan, _atr


def add_gap_fill_risk(df, symbol_col='symbol', date_col='date',
                      open_col='open', high_col='high', low_col='low',
                      close_col='close', volume_col='volume'):
    """
    Add gap fill risk metrics to each row:
      - filled_same_day: bool — did the gap fill on the same session?
      - fill_distance_pct: float — how far price traveled toward filling (0-1+)
        1.0 = fully filled, 0.0 = never retraced, >1.0 = overshot fill level
      - gap_fill_risk: str — 'low', 'medium', 'high' based on historical fill rate
        and intraday fill behavior
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])
    out = []

    for sym, g in df.groupby(symbol_col):
        g = g.copy().reset_index(drop=True)
        prev_close = g[close_col].shift(1)
        gap_pct = (g[open_col] / prev_close) - 1

        is_gap_up = gap_pct > 0.02
        is_gap_down = gap_pct < -0.02

        # ── Filled same day ──
        # Gap up fills if low touches prev_close; gap down fills if high touches prev_close
        filled_up = is_gap_up & (g[low_col] <= prev_close)
        filled_down = is_gap_down & (g[high_col] >= prev_close)
        g['filled_same_day'] = filled_up | filled_down

        # ── Fill distance (0 to 1+) ──
        # For gap ups: how far did the low retrace toward prev_close?
        # fill_distance = (open - low) / (open - prev_close)
        gap_up_dist = np.where(
            is_gap_up & (g[open_col] != prev_close),
            (g[open_col] - g[low_col]) / (g[open_col] - prev_close),
            np.nan
        )
        # For gap downs: how far did the high retrace toward prev_close?
        gap_down_dist = np.where(
            is_gap_down & (prev_close != g[open_col]),
            (g[high_col] - g[open_col]) / (prev_close - g[open_col]),
            np.nan
        )
        g['fill_distance_pct'] = np.where(
            is_gap_up, gap_up_dist,
            np.where(is_gap_down, gap_down_dist, np.nan)
        )
        g['fill_distance_pct'] = pd.to_numeric(g['fill_distance_pct'], errors='coerce').round(3)

        # ── Historical fill rate (rolling 20 gaps) ──
        # Count how many of the last 20 gaps for this symbol filled same-day
        has_gap = is_gap_up | is_gap_down
        gap_rows = g[has_gap].copy()
        if len(gap_rows) > 0:
            gap_rows['_fill_rate_20'] = gap_rows['filled_same_day'].rolling(20, min_periods=3).mean()
            g = g.merge(
                gap_rows[['_fill_rate_20']],
                left_index=True, right_index=True, how='left'
            )
            g['_fill_rate_20'] = g['_fill_rate_20'].ffill()
        else:
            g['_fill_rate_20'] = np.nan

        # ── Gap fill risk classification ──
        conditions = [
            # High risk: >60% historical fill rate OR filled same day with small gap
            (g['_fill_rate_20'] > 0.6) | (g['filled_same_day'] & (gap_pct.abs() < 0.04)),
            # Medium risk: 30-60% fill rate OR partial fill (>50% distance)
            (g['_fill_rate_20'] > 0.3) | (g['fill_distance_pct'] > 0.5),
        ]
        choices = ['high', 'medium']
        g['gap_fill_risk'] = np.select(conditions, choices, default='low')

        # Only assign risk to actual gap days
        g['filled_same_day'] = g['filled_same_day'].astype(object)
        g.loc[~has_gap, 'gap_fill_risk'] = 'n/a'
        g.loc[~has_gap, 'filled_same_day'] = None
        g.loc[~has_gap, 'fill_distance_pct'] = np.nan

        g.drop(columns=['_fill_rate_20'], inplace=True)
        g['symbol'] = sym
        out.append(g)

    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    print("Patch 1: Adding gap fill risk metrics...")
    df = pd.read_csv('ohlcv.csv')
    result = breakaway_gap_scan(df)
    result_with_fill = add_gap_fill_risk(result)

    gap_rows = result_with_fill[result_with_fill['gap_fill_risk'] != 'n/a']
    print(f"\nGap days found: {len(gap_rows)}")
    print(f"Fill risk distribution:")
    print(gap_rows['gap_fill_risk'].value_counts().to_string())
    print(f"\nSame-day fill rate: {gap_rows['filled_same_day'].mean():.1%}")
    print(f"\nSample output:")
    print(gap_rows[['symbol', 'date', 'gap_pct', 'filled_same_day', 'fill_distance_pct', 'gap_fill_risk']].head(15).to_string(index=False))

    result_with_fill.to_csv('breakaway_gap_watchlist.csv', index=False)
    print("\nSaved to breakaway_gap_watchlist.csv")
