"""
Patch 2: Post-Earnings Gap Flag
Adds: post_earnings_flag_scan()

Flags gaps that occur within 1 trading day of an earnings date.
Earnings gaps behave differently — they have lower fill rates
and higher continuation probability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from breakaway_gap_scan import breakaway_gap_scan


# ── Known earnings calendar (approximate quarterly dates) ──
# In production, pull from yfinance t.calendar or an earnings API

def _get_earnings_dates_yfinance(symbols):
    """Try to fetch earnings dates from yfinance. Returns dict: {symbol: [date, ...]}"""
    try:
        import yfinance as yf
        earnings = {}
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                cal = t.calendar
                if cal is not None and not cal.empty:
                    dates = []
                    if hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                        dates = pd.to_datetime(cal['Earnings Date']).tolist()
                    elif hasattr(cal, 'index'):
                        dates = pd.to_datetime(cal.index).tolist()
                    earnings[sym] = dates
            except Exception:
                continue
        return earnings
    except ImportError:
        return {}


def post_earnings_flag_scan(df, symbol_col='symbol', date_col='date',
                            open_col='open', close_col='close',
                            earnings_dates=None, window_days=2):
    """
    Flag rows where a gap occurs within `window_days` of an earnings report.

    Parameters:
        df: DataFrame with OHLCV + gap columns (output of breakaway_gap_scan)
        earnings_dates: dict {symbol: [datetime, ...]} — if None, attempts yfinance fetch
        window_days: int — trading days around earnings to flag

    Adds columns:
        - is_earnings_gap: bool — gap occurred near earnings
        - earnings_date_nearest: date — the nearest earnings date
        - days_from_earnings: int — trading days from nearest earnings
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    symbols = df[symbol_col].unique().tolist()

    # Fetch earnings dates if not provided
    if earnings_dates is None:
        print(f"Fetching earnings dates for {len(symbols)} symbols...")
        earnings_dates = _get_earnings_dates_yfinance(symbols)
        print(f"Got earnings data for {len(earnings_dates)} symbols")

    df['is_earnings_gap'] = False
    df['earnings_date_nearest'] = pd.NaT
    df['days_from_earnings'] = np.nan

    for sym in symbols:
        sym_mask = df[symbol_col] == sym
        sym_dates = df.loc[sym_mask, date_col]

        edates = earnings_dates.get(sym, [])
        if not edates:
            continue

        edates = pd.to_datetime(edates)

        for idx in df[sym_mask].index:
            row_date = df.loc[idx, date_col]
            # Find nearest earnings date
            diffs = (edates - row_date).days
            abs_diffs = abs(diffs)
            nearest_idx = abs_diffs.argmin()
            nearest_date = edates[nearest_idx]
            days_diff = int(diffs[nearest_idx])

            df.loc[idx, 'earnings_date_nearest'] = nearest_date
            df.loc[idx, 'days_from_earnings'] = abs(days_diff)

            # Flag if within window AND there's a gap
            has_gap = df.loc[idx, 'gap_up'] if 'gap_up' in df.columns else False
            has_gap = has_gap or (df.loc[idx, 'gap_down'] if 'gap_down' in df.columns else False)

            if abs(days_diff) <= window_days and has_gap:
                df.loc[idx, 'is_earnings_gap'] = True

    return df


if __name__ == '__main__':
    print("Patch 2: Adding post-earnings gap flags...")
    df = pd.read_csv('ohlcv.csv')
    result = breakaway_gap_scan(df)
    result = post_earnings_flag_scan(result)

    earnings_gaps = result[result['is_earnings_gap']]
    print(f"\nEarnings gaps found: {len(earnings_gaps)}")
    if len(earnings_gaps) > 0:
        print(earnings_gaps[['symbol', 'date', 'gap_pct', 'signal', 'days_from_earnings']].head(15).to_string(index=False))

    result.to_csv('breakaway_gap_watchlist.csv', index=False)
    print("\nSaved to breakaway_gap_watchlist.csv")
