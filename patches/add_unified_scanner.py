"""
Patch 4: Climax Top Scanner + Unified Event Scanner
Adds: climax_top_scan() + unified_event_scan()

climax_top_scan() implements O'Neil/Livermore climax top detection:
  - Extended prior run (25%+ in 5-15 days or 50%+ in 10-15 days)
  - Exhaustion behavior (7+ up days in 8, or 8+ in 10, or widest weekly range)
  - Confirmation (exhaustion gap, volume spike, or heavy up-week volume)

unified_event_scan() chains all patches into a single pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from breakaway_gap_scan import breakaway_gap_scan
from patches.add_gap_fill_risk import add_gap_fill_risk
from patches.add_post_earnings_flag import post_earnings_flag_scan
from patches.add_composite_event import add_composite_event


def climax_top_scan(df, symbol_col='symbol', date_col='date',
                    open_col='open', high_col='high', low_col='low',
                    close_col='close', volume_col='volume'):
    """
    Detect climax top sell signals using O'Neil/Livermore criteria.

    A climax top requires ALL of:
      1. Extended prior run (price acceleration)
      2. Exhaustion buying behavior (consecutive up days, wide range)
      3. Confirmation signal (gap, volume spike, heavy weekly volume)

    Adds columns:
      - climax_prior_run_extended: bool
      - climax_exhaustion: bool
      - climax_confirmation: bool
      - climax_top: bool — all 3 conditions met
      - trendline_break_heavy_vol: bool — 20d SMA break on heavy volume
      - sell_signal: bool — climax_top OR trendline_break_heavy_vol
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])
    out = []

    for sym, g in df.groupby(symbol_col):
        g = g.copy().reset_index(drop=True)
        n = len(g)

        close = g[close_col].values
        high = g[high_col].values
        low = g[low_col].values
        opn = g[open_col].values
        vol = g[volume_col].values
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        # ── 1. Prior Run Extended ──
        # 25%+ gain in last 5-15 days OR 50%+ gain in last 10-15 days
        pct_5d = pd.Series(close).pct_change(5).values
        pct_10d = pd.Series(close).pct_change(10).values
        pct_15d = pd.Series(close).pct_change(15).values

        # Best gain in the 5-15 day window
        pct_5_to_15 = np.maximum(pct_5d, np.maximum(pct_10d, pct_15d))
        # Best gain in the 10-15 day window
        pct_10_to_15 = np.maximum(pct_10d, pct_15d)

        prior_run_extended = (pct_5_to_15 >= 0.25) | (pct_10_to_15 >= 0.50)

        # ── 2. Exhaustion Buying ──
        up_day = close > prev_close
        up_days_8 = pd.Series(up_day.astype(int)).rolling(8).sum().values
        up_days_10 = pd.Series(up_day.astype(int)).rolling(10).sum().values

        # Widest weekly range of the move (rolling 5-day range vs 20-day max)
        range_5d = pd.Series(high).rolling(5).max().values - pd.Series(low).rolling(5).min().values
        range_20d_max = pd.Series(range_5d).rolling(20).max().values
        widest_weekly = range_5d >= range_20d_max * 0.95  # Within 5% of max

        exhaustion = (up_days_8 >= 7) | (up_days_10 >= 8) | widest_weekly

        # ── 3. Confirmation ──
        # Exhaustion gap: gap up on heavy volume after extended run
        gap_up = opn > prev_close * 1.01  # >1% gap up
        vol_ma_50 = pd.Series(vol, dtype=float).rolling(50).mean().values
        vol_spike = vol >= 1.5 * vol_ma_50

        # Heavy volume on up weeks (rolling 5-day volume vs 10-week avg)
        vol_5d = pd.Series(vol, dtype=float).rolling(5).sum().values
        vol_50d_avg = pd.Series(vol, dtype=float).rolling(50).mean().values * 5
        heavy_up_weeks = vol_5d >= 1.3 * vol_50d_avg

        confirmation = gap_up | vol_spike | heavy_up_weeks

        # ── Climax Top ──
        climax_top = prior_run_extended & exhaustion & confirmation

        # ── Trendline Break on Heavy Volume ──
        sma_20 = pd.Series(close).rolling(20).mean().values
        broke_below_sma = (close < sma_20) & (prev_close >= sma_20)
        trendline_break = broke_below_sma & vol_spike

        # ── Sell Signal ──
        sell_signal = climax_top | trendline_break

        g['climax_prior_run_extended'] = prior_run_extended
        g['climax_exhaustion'] = exhaustion
        g['climax_confirmation'] = confirmation
        g['climax_top'] = climax_top
        g['trendline_break_heavy_vol'] = trendline_break
        g['sell_signal'] = sell_signal
        g['symbol'] = sym
        out.append(g)

    return pd.concat(out, ignore_index=True)


def unified_event_scan(df, symbol_col='symbol', date_col='date',
                       open_col='open', high_col='high', low_col='low',
                       close_col='close', volume_col='volume',
                       earnings_dates=None):
    """
    Run the full event detection pipeline:
      1. Breakaway gap scan
      2. Gap fill risk analysis
      3. Post-earnings flag
      4. Climax top detection
      5. Composite event labeling

    Returns a fully annotated DataFrame sorted by event_score.
    """
    print("Step 1/5: Breakaway gap scan...")
    result = breakaway_gap_scan(df, symbol_col, date_col, open_col,
                                high_col, low_col, close_col, volume_col)

    print("Step 2/5: Gap fill risk analysis...")
    result = add_gap_fill_risk(result, symbol_col, date_col, open_col,
                               high_col, low_col, close_col, volume_col)

    print("Step 3/5: Post-earnings flag...")
    result = post_earnings_flag_scan(result, symbol_col, date_col,
                                     open_col, close_col,
                                     earnings_dates=earnings_dates)

    print("Step 4/5: Climax top detection...")
    result = climax_top_scan(result, symbol_col, date_col, open_col,
                             high_col, low_col, close_col, volume_col)

    print("Step 5/5: Composite event labeling...")
    # Upgrade event labels for climax tops
    result = add_composite_event(result, symbol_col, date_col)

    # Override event for climax tops
    climax_mask = result['climax_top'].fillna(False)
    result.loc[climax_mask, 'event_code'] = 'climax_top'
    result.loc[climax_mask, 'event_label'] = 'Climax Top — Sell Signal'
    result.loc[climax_mask, 'event_score'] = result.loc[climax_mask, 'event_score'].clip(lower=7.0)

    trendline_mask = result['trendline_break_heavy_vol'].fillna(False) & ~climax_mask
    result.loc[trendline_mask, 'event_code'] = 'trendline_break'
    result.loc[trendline_mask, 'event_label'] = 'Trendline Break — Heavy Volume'
    result.loc[trendline_mask, 'event_score'] = result.loc[trendline_mask, 'event_score'].clip(lower=5.0)

    result = result.sort_values(['event_score', date_col], ascending=[False, False])

    print(f"\nPipeline complete: {len(result)} rows, "
          f"{(result['event_code'] != 'none').sum()} events detected")

    return result


if __name__ == '__main__':
    print("Patch 4: Running unified event scanner...")
    df = pd.read_csv('ohlcv.csv')
    result = unified_event_scan(df)

    events = result[result['event_code'] != 'none']
    print(f"\n{'='*60}")
    print(f"EVENT SUMMARY")
    print(f"{'='*60}")
    print(events['event_label'].value_counts().to_string())

    climax = result[result['climax_top'].fillna(False)]
    if len(climax) > 0:
        print(f"\nWARNING:  CLIMAX TOPS DETECTED: {len(climax)}")
        print(climax[['symbol', 'date', 'close', 'event_score']].to_string(index=False))

    sells = result[result['sell_signal'].fillna(False)]
    if len(sells) > 0:
        print(f"\n>> SELL SIGNALS: {len(sells)}")
        print(sells[['symbol', 'date', 'close', 'event_label', 'event_score']].head(20).to_string(index=False))

    result.to_csv('unified_event_watchlist.csv', index=False)
    print("\nSaved to unified_event_watchlist.csv")
