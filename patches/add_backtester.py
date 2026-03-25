"""
Patch 8: Backtester
Adds: backtest_events() and --mode backtest to CLI

Runs the full stage-enriched pipeline, then computes forward returns
at 5, 10, 20, 40, and 60 trading days for every flagged event.

Outputs:
  1. Performance by event_label
  2. Performance by event_label x stage cross
  3. Highest-edge setups ranked by 20-day expectancy
  4. Statistical significance summary (t-test vs zero)
  5. MFE/MAE (max favorable/adverse excursion) per signal
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy import stats

from patches.add_stage_enriched import stage_enriched_scan

logger = logging.getLogger("backtester")

FORWARD_WINDOWS = [5, 10, 20, 40, 60]


def compute_forward_returns(df, ohlcv_df, symbol_col='symbol', date_col='date',
                            close_col='close'):
    """
    For each event row, compute forward returns at 5/10/20/40/60 day horizons.
    Also computes MFE (max favorable excursion) and MAE (max adverse excursion)
    over the 60-day forward window.

    MFE = max gain from entry before exit (best unrealized P&L)
    MAE = max drawdown from entry before exit (worst unrealized P&L)
    """
    df = df.copy()
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df[date_col] = pd.to_datetime(ohlcv_df[date_col])
    df[date_col] = pd.to_datetime(df[date_col])

    # Build forward price lookup per symbol
    # For each (symbol, date), store array of next 60 closes
    forward_cache = {}
    for sym, g in ohlcv_df.sort_values(date_col).groupby(symbol_col):
        closes = g[close_col].values
        highs = g['high'].values if 'high' in g.columns else closes
        lows = g['low'].values if 'low' in g.columns else closes
        dates = g[date_col].values
        date_to_idx = {d: i for i, d in enumerate(dates)}
        forward_cache[sym] = {
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'dates': dates,
            'date_to_idx': date_to_idx,
        }

    # Initialize columns
    for w in FORWARD_WINDOWS:
        df[f'fwd_{w}d_ret'] = np.nan
    df['mfe_60d'] = np.nan  # Max favorable excursion (60 day window)
    df['mae_60d'] = np.nan  # Max adverse excursion (60 day window)
    df['mfe_20d'] = np.nan
    df['mae_20d'] = np.nan

    for idx in df.index:
        sym = df.loc[idx, symbol_col]
        row_date = df.loc[idx, date_col]

        cache = forward_cache.get(sym)
        if cache is None:
            continue

        date_idx = cache['date_to_idx'].get(row_date)
        if date_idx is None:
            # Try nearest date match
            diffs = np.abs(cache['dates'] - row_date)
            nearest = np.argmin(diffs)
            if diffs[nearest] > np.timedelta64(3, 'D'):
                continue
            date_idx = nearest

        entry_price = cache['closes'][date_idx]
        if entry_price == 0 or np.isnan(entry_price):
            continue

        n = len(cache['closes'])
        direction = df.loc[idx, 'direction'] if 'direction' in df.columns else 'bullish'
        is_bearish = direction == 'bearish' or (
            df.loc[idx, 'sell_signal'] if 'sell_signal' in df.columns else False
        )

        # Forward returns at each window
        for w in FORWARD_WINDOWS:
            fwd_idx = date_idx + w
            if fwd_idx < n:
                fwd_price = cache['closes'][fwd_idx]
                ret = (fwd_price / entry_price) - 1
                # For bearish signals, invert return (short profit)
                if is_bearish:
                    ret = -ret
                df.loc[idx, f'fwd_{w}d_ret'] = round(ret * 100, 3)

        # MFE/MAE over 60-day window
        max_window = min(date_idx + 61, n)
        if max_window > date_idx + 1:
            fwd_highs = cache['highs'][date_idx + 1:max_window]
            fwd_lows = cache['lows'][date_idx + 1:max_window]

            if is_bearish:
                # Bearish: favorable = price goes down, adverse = price goes up
                mfe = (entry_price - np.min(fwd_lows)) / entry_price * 100
                mae = (np.max(fwd_highs) - entry_price) / entry_price * 100
            else:
                # Bullish: favorable = price goes up, adverse = price goes down
                mfe = (np.max(fwd_highs) - entry_price) / entry_price * 100
                mae = (entry_price - np.min(fwd_lows)) / entry_price * 100

            df.loc[idx, 'mfe_60d'] = round(mfe, 2)
            df.loc[idx, 'mae_60d'] = round(mae, 2)

        # MFE/MAE over 20-day window
        max_window_20 = min(date_idx + 21, n)
        if max_window_20 > date_idx + 1:
            fwd_highs_20 = cache['highs'][date_idx + 1:max_window_20]
            fwd_lows_20 = cache['lows'][date_idx + 1:max_window_20]

            if is_bearish:
                mfe_20 = (entry_price - np.min(fwd_lows_20)) / entry_price * 100
                mae_20 = (np.max(fwd_highs_20) - entry_price) / entry_price * 100
            else:
                mfe_20 = (np.max(fwd_highs_20) - entry_price) / entry_price * 100
                mae_20 = (entry_price - np.min(fwd_lows_20)) / entry_price * 100

            df.loc[idx, 'mfe_20d'] = round(mfe_20, 2)
            df.loc[idx, 'mae_20d'] = round(mae_20, 2)

    return df


def table_by_event_label(df, min_samples=3):
    """Table 1: Performance by event_label."""
    events = df[df.get('event_code', pd.Series('none')) != 'none'].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for label, g in events.groupby('event_label'):
        if len(g) < min_samples:
            continue
        row = {'event_label': label, 'count': len(g)}
        for w in FORWARD_WINDOWS:
            col = f'fwd_{w}d_ret'
            valid = g[col].dropna()
            if len(valid) > 0:
                row[f'avg_{w}d'] = round(valid.mean(), 2)
                row[f'med_{w}d'] = round(valid.median(), 2)
                row[f'win_{w}d'] = round((valid > 0).mean() * 100, 1)
            else:
                row[f'avg_{w}d'] = None
                row[f'med_{w}d'] = None
                row[f'win_{w}d'] = None

        # MFE/MAE
        mfe_valid = g['mfe_60d'].dropna()
        mae_valid = g['mae_60d'].dropna()
        row['avg_mfe_60d'] = round(mfe_valid.mean(), 2) if len(mfe_valid) > 0 else None
        row['avg_mae_60d'] = round(mae_valid.mean(), 2) if len(mae_valid) > 0 else None
        row['edge_ratio'] = round(mfe_valid.mean() / mae_valid.mean(), 2) if len(mae_valid) > 0 and mae_valid.mean() > 0 else None

        rows.append(row)

    return pd.DataFrame(rows).sort_values('avg_20d', ascending=False, na_position='last')


def table_by_label_x_stage(df, min_samples=2):
    """Table 2: Performance by event_label x Weinstein stage cross."""
    events = df[df.get('event_code', pd.Series('none')) != 'none'].copy()
    if events.empty or 'w_stage_label' not in events.columns:
        return pd.DataFrame()

    rows = []
    for (label, stage), g in events.groupby(['event_label', 'w_stage_label']):
        if len(g) < min_samples:
            continue
        row = {
            'event_label': label,
            'stage': stage,
            'count': len(g),
        }
        for w in [10, 20, 40]:
            col = f'fwd_{w}d_ret'
            valid = g[col].dropna()
            if len(valid) > 0:
                row[f'avg_{w}d'] = round(valid.mean(), 2)
                row[f'win_{w}d'] = round((valid > 0).mean() * 100, 1)

        mfe = g['mfe_20d'].dropna()
        mae = g['mae_20d'].dropna()
        row['avg_mfe_20d'] = round(mfe.mean(), 2) if len(mfe) > 0 else None
        row['avg_mae_20d'] = round(mae.mean(), 2) if len(mae) > 0 else None

        rows.append(row)

    result = pd.DataFrame(rows)
    if 'avg_20d' in result.columns:
        result = result.sort_values('avg_20d', ascending=False, na_position='last')
    return result


def table_highest_edge(df, min_samples=3):
    """Table 3: Highest-edge setups ranked by 20-day expectancy."""
    events = df[df.get('event_code', pd.Series('none')) != 'none'].copy()
    if events.empty:
        return pd.DataFrame()

    # Group by event_code + stage for granular edge
    group_cols = ['event_code']
    if 'w_stage' in events.columns:
        group_cols.append('w_stage')
    if 'iv_regime' in events.columns:
        group_cols.append('iv_regime')

    rows = []
    for keys, g in events.groupby(group_cols):
        if len(g) < min_samples:
            continue

        if isinstance(keys, tuple):
            row = dict(zip(group_cols, keys))
        else:
            row = {group_cols[0]: keys}

        row['count'] = len(g)

        ret_20 = g['fwd_20d_ret'].dropna()
        if len(ret_20) == 0:
            continue

        row['avg_20d_ret'] = round(ret_20.mean(), 2)
        row['med_20d_ret'] = round(ret_20.median(), 2)
        row['win_rate_20d'] = round((ret_20 > 0).mean() * 100, 1)
        row['std_20d'] = round(ret_20.std(), 2)

        # Expectancy = avg_win * win_rate - avg_loss * loss_rate
        wins = ret_20[ret_20 > 0]
        losses = ret_20[ret_20 <= 0]
        win_rate = len(wins) / len(ret_20) if len(ret_20) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        row['expectancy_20d'] = round(avg_win * win_rate - avg_loss * (1 - win_rate), 3)

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.01
        row['profit_factor'] = round(gross_profit / gross_loss, 2)

        # MFE/MAE edge ratio
        mfe = g['mfe_20d'].dropna()
        mae = g['mae_20d'].dropna()
        if len(mae) > 0 and mae.mean() > 0:
            row['edge_ratio_20d'] = round(mfe.mean() / mae.mean(), 2)
        else:
            row['edge_ratio_20d'] = None

        rows.append(row)

    result = pd.DataFrame(rows)
    if 'expectancy_20d' in result.columns:
        result = result.sort_values('expectancy_20d', ascending=False)
    return result


def table_significance(df, min_samples=5):
    """Table 4: Statistical significance — t-test of forward returns vs zero."""
    events = df[df.get('event_code', pd.Series('none')) != 'none'].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for label, g in events.groupby('event_label'):
        row = {'event_label': label, 'count': len(g)}

        for w in FORWARD_WINDOWS:
            col = f'fwd_{w}d_ret'
            valid = g[col].dropna()

            if len(valid) < min_samples:
                row[f't_stat_{w}d'] = None
                row[f'p_value_{w}d'] = None
                row[f'sig_{w}d'] = ''
                continue

            t_stat, p_value = stats.ttest_1samp(valid, 0)
            row[f't_stat_{w}d'] = round(t_stat, 3)
            row[f'p_value_{w}d'] = round(p_value, 4)

            # Significance markers
            if p_value < 0.01:
                row[f'sig_{w}d'] = '***'
            elif p_value < 0.05:
                row[f'sig_{w}d'] = '**'
            elif p_value < 0.10:
                row[f'sig_{w}d'] = '*'
            else:
                row[f'sig_{w}d'] = ''

        rows.append(row)

    return pd.DataFrame(rows)


def backtest_events(df, symbol_col='symbol', date_col='date',
                    open_col='open', high_col='high', low_col='low',
                    close_col='close', volume_col='volume'):
    """
    Full backtest pipeline:
      1. Run stage-enriched scan
      2. Compute forward returns + MFE/MAE
      3. Generate all 4 analysis tables

    Returns:
      (result_df, tables_dict, regime, stages)
    """
    # ── Step 1: Stage-enriched scan ──
    result, regime, stages = stage_enriched_scan(
        df, symbol_col, date_col, open_col,
        high_col, low_col, close_col, volume_col
    )

    # ── Step 2: Forward returns + MFE/MAE ──
    print("\n" + "=" * 60)
    print("  BACKTESTER")
    print("=" * 60)
    print("Computing forward returns (5/10/20/40/60 day)...")

    result = compute_forward_returns(result, df, symbol_col, date_col, close_col)

    events = result[result.get('event_code', pd.Series('none')) != 'none']
    has_returns = events['fwd_20d_ret'].notna().sum()
    print(f"Events with forward data: {has_returns} / {len(events)}")

    # ── Step 3: Analysis tables ──
    print("\nGenerating analysis tables...")

    t1 = table_by_event_label(result)
    t2 = table_by_label_x_stage(result)
    t3 = table_highest_edge(result)
    t4 = table_significance(result)

    tables = {
        'by_event_label': t1,
        'label_x_stage': t2,
        'highest_edge': t3,
        'significance': t4,
    }

    # ── Print tables ──
    print("\n" + "=" * 60)
    print("  TABLE 1: Performance by Event Label")
    print("=" * 60)
    if not t1.empty:
        display_cols = [c for c in ['event_label', 'count', 'avg_5d', 'avg_10d', 'avg_20d',
                                     'avg_40d', 'win_20d', 'avg_mfe_60d', 'avg_mae_60d', 'edge_ratio']
                        if c in t1.columns]
        print(t1[display_cols].to_string(index=False))
    else:
        print("  (no data)")

    print("\n" + "=" * 60)
    print("  TABLE 2: Performance by Event Label x Stage")
    print("=" * 60)
    if not t2.empty:
        display_cols = [c for c in ['event_label', 'stage', 'count', 'avg_10d', 'avg_20d',
                                     'win_20d', 'avg_mfe_20d', 'avg_mae_20d']
                        if c in t2.columns]
        print(t2[display_cols].head(25).to_string(index=False))
    else:
        print("  (no data)")

    print("\n" + "=" * 60)
    print("  TABLE 3: Highest-Edge Setups (by 20d Expectancy)")
    print("=" * 60)
    if not t3.empty:
        display_cols = [c for c in ['event_code', 'w_stage', 'iv_regime', 'count',
                                     'avg_20d_ret', 'win_rate_20d', 'expectancy_20d',
                                     'profit_factor', 'edge_ratio_20d']
                        if c in t3.columns]
        print(t3[display_cols].head(20).to_string(index=False))
    else:
        print("  (no data)")

    print("\n" + "=" * 60)
    print("  TABLE 4: Statistical Significance (t-test vs 0)")
    print("=" * 60)
    if not t4.empty:
        display_cols = [c for c in ['event_label', 'count',
                                     't_stat_10d', 'p_value_10d', 'sig_10d',
                                     't_stat_20d', 'p_value_20d', 'sig_20d',
                                     't_stat_40d', 'p_value_40d', 'sig_40d']
                        if c in t4.columns]
        print(t4[display_cols].to_string(index=False))
        print("\n  Significance: *** p<0.01, ** p<0.05, * p<0.10")
    else:
        print("  (no data)")

    # ── MFE/MAE summary ──
    print("\n" + "=" * 60)
    print("  MFE / MAE SUMMARY (60-day window)")
    print("=" * 60)
    mfe_mae = events[['event_label', 'mfe_60d', 'mae_60d', 'mfe_20d', 'mae_20d']].copy()
    mfe_mae = mfe_mae.dropna(subset=['mfe_60d'])
    if not mfe_mae.empty:
        summary = mfe_mae.groupby('event_label').agg(
            count=('mfe_60d', 'count'),
            avg_mfe_60d=('mfe_60d', 'mean'),
            avg_mae_60d=('mae_60d', 'mean'),
            avg_mfe_20d=('mfe_20d', 'mean'),
            avg_mae_20d=('mae_20d', 'mean'),
        ).round(2)
        summary['edge_60d'] = (summary['avg_mfe_60d'] / summary['avg_mae_60d'].clip(lower=0.01)).round(2)
        summary['edge_20d'] = (summary['avg_mfe_20d'] / summary['avg_mae_20d'].clip(lower=0.01)).round(2)
        summary = summary.sort_values('edge_60d', ascending=False)
        print(summary.to_string())
        print("\n  Edge ratio > 1.5 = favorable risk/reward")
    else:
        print("  (no MFE/MAE data)")

    print("\n" + "=" * 60)

    return result, tables, regime, stages


if __name__ == '__main__':
    print("Patch 8: Running backtest...")
    df = pd.read_csv('ohlcv.csv')
    result, tables, regime, stages = backtest_events(df)

    result.to_csv('backtest_results.csv', index=False)
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(f'backtest_{name}.csv', index=False)
    print("\nAll backtest outputs saved.")
