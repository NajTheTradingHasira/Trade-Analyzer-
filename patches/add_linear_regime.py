"""
Patch 19: Linear Regime Alerts
Detects trend regime changes from the linear tracker output.

linear_regime_alerts():
  - Trend acceleration: R2 improving + slope increasing over 5 days
  - Trend collapse: R2 dropping below 0.5 with sharp decline
  - linear_regime: 'accelerating' / 'collapsing' / 'neutral'
  - linear_regime_alert: True on regime change days

Key insight:
  Collapsing R2 = the trend is breaking down. In Stage 2, this is
  the first warning of Stage 3 transition (before price confirms).
  Accelerating R2 in Stage 1B = the base is tightening, breakout imminent.

  Combines with linear_tracker to produce regime-aware trend analysis.

CLI: --mode linear_regime
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from patches.add_linear_tracker import linear_tracker

logger = logging.getLogger("linear_regime")


def linear_regime_alerts(df, symbol_col='symbol', date_col='date'):
    """
    Detect trend regime changes from linear tracker data.

    Adds columns:
      lin_r2_chg_5d       — 5-day change in R2
      lin_slope_chg_5d    — 5-day change in slope
      trend_collapse      — bool, R2 10d avg < 0.5 AND R2 dropped > 0.15 in 5d
      trend_accel         — bool, R2 gained > 0.10 in 5d AND slope also improved
      linear_regime       — 'accelerating', 'collapsing', or 'neutral'
      linear_regime_alert — bool, True when regime is not neutral
    """
    d = df.copy()

    if 'lin_r2' not in d.columns or 'lin_slope' not in d.columns:
        logger.warning("Missing lin_r2 or lin_slope — run linear_tracker first")
        d['linear_regime'] = 'neutral'
        d['linear_regime_alert'] = False
        return d

    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([symbol_col, date_col])

    out = []
    for sym, g in d.groupby(symbol_col):
        g = g.copy()

        # 5-day changes in R2 and slope
        g['lin_r2_chg_5d'] = g['lin_r2'] - g['lin_r2'].shift(5)
        g['lin_slope_chg_5d'] = g['lin_slope'] - g['lin_slope'].shift(5)

        # 10-day rolling average R2
        g['lin_r2_10d_avg'] = g['lin_r2'].rolling(10, min_periods=3).mean()

        # Trend collapse: R2 average < 0.5 AND R2 dropped sharply
        g['trend_collapse'] = (
            (g['lin_r2_10d_avg'] < 0.5) &
            (g['lin_r2_chg_5d'] < -0.15)
        ).fillna(False)

        # Trend acceleration: R2 improving AND slope also improving
        g['trend_accel'] = (
            (g['lin_r2_chg_5d'] > 0.10) &
            (g['lin_slope_chg_5d'] > 0)
        ).fillna(False)

        # Regime label
        g['linear_regime'] = 'neutral'
        g.loc[g['trend_accel'], 'linear_regime'] = 'accelerating'
        g.loc[g['trend_collapse'], 'linear_regime'] = 'collapsing'

        # Alert flag
        g['linear_regime_alert'] = g['linear_regime'].isin(['accelerating', 'collapsing'])

        out.append(g)

    return pd.concat(out, ignore_index=True)


def print_regime_summary(latest, symbol_col='symbol'):
    """Print regime alerts from latest snapshot."""
    accel = latest[latest.get('linear_regime', pd.Series('neutral')) == 'accelerating']
    collapse = latest[latest.get('linear_regime', pd.Series('neutral')) == 'collapsing']
    neutral = latest[latest.get('linear_regime', pd.Series('neutral')) == 'neutral']

    print(f"\n{'='*80}")
    print(f"  LINEAR REGIME ALERTS")
    print(f"{'='*80}")

    if len(accel) > 0:
        print(f"\n  ACCELERATING ({len(accel)}) — trend tightening, breakout potential:")
        for _, r in accel.iterrows():
            print(f"    {r.get(symbol_col, '?'):7s}  "
                  f"R2={r.get('lin_r2', 0):.3f}  "
                  f"R2 chg={r.get('lin_r2_chg_5d', 0):+.3f}  "
                  f"slope={r.get('lin_slope', 0):+.3f}  "
                  f"slope chg={r.get('lin_slope_chg_5d', 0):+.3f}")
    else:
        print(f"\n  ACCELERATING: none")

    if len(collapse) > 0:
        print(f"\n  COLLAPSING ({len(collapse)}) — trend breaking down, Stage transition warning:")
        for _, r in collapse.iterrows():
            print(f"    {r.get(symbol_col, '?'):7s}  "
                  f"R2={r.get('lin_r2', 0):.3f}  "
                  f"R2 chg={r.get('lin_r2_chg_5d', 0):+.3f}  "
                  f"slope={r.get('lin_slope', 0):+.3f}  "
                  f"R2 10d avg={r.get('lin_r2_10d_avg', 0):.3f}")
    else:
        print(f"\n  COLLAPSING: none")

    print(f"\n  NEUTRAL: {len(neutral)} tickers")

    # Stage interpretation
    print(f"\n  Interpretation guide:")
    print(f"    Accelerating + Stage 1  = base tightening, breakout imminent")
    print(f"    Accelerating + Stage 4  = bear trend intensifying")
    print(f"    Collapsing   + Stage 2  = first warning of Stage 3 (before price)")
    print(f"    Collapsing   + Stage 4  = bear trend exhausting, possible Stage 1")

    print(f"{'='*80}")


def run_linear_regime_mode(ohlcv_df, outdir='.', stamp=None, lookback=126):
    """Full linear regime pipeline for CLI mode."""
    if stamp is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "=" * 60)
    print(f"  LINEAR REGIME ANALYSIS (lookback={lookback} days)")
    print("=" * 60)

    # Step 1: Run linear tracker to get full timeseries
    full, latest, csv_path, html_path = linear_tracker(
        ohlcv_df, lookback=lookback, outdir=outdir, stem=f'linear_regime_{stamp}'
    )

    if full.empty:
        print("No data to analyze.")
        return pd.DataFrame(), pd.DataFrame(), None

    # Step 2: Apply regime detection on full timeseries
    full_with_regime = linear_regime_alerts(full)

    # Step 3: Get latest snapshot with regime info
    latest_regime = full_with_regime.sort_values('date').groupby('symbol', as_index=False).tail(1)
    latest_regime = latest_regime.sort_values('linear_rank' if 'linear_rank' in latest_regime.columns else 'linear_strength', ascending=False)

    # Step 4: Print tracker table
    from patches.add_linear_tracker import print_linear_tracker_table
    print_linear_tracker_table(latest_regime)

    # Step 5: Print regime alerts
    print_regime_summary(latest_regime)

    # Save
    regime_csv = Path(outdir) / f'linear_regime_{stamp}.csv'
    display_cols = [c for c in [
        'symbol', 'date', 'close', 'lin_slope', 'lin_r2', 'lin_pos',
        'linear_strength', 'linear_rank', 'linear_flag', 'linear_label',
        'lin_r2_chg_5d', 'lin_slope_chg_5d', 'lin_r2_10d_avg',
        'linear_regime', 'linear_regime_alert',
    ] if c in latest_regime.columns]
    latest_regime[display_cols].to_csv(regime_csv, index=False)

    print(f"\n  Outputs:")
    print(f"    Regime CSV: {regime_csv}")
    if html_path:
        print(f"    Chart: {html_path}")

    return full_with_regime, latest_regime, regime_csv


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 19: Testing linear regime alerts...")
    df = pd.read_csv('ohlcv.csv')
    full, latest, csv = run_linear_regime_mode(df)
