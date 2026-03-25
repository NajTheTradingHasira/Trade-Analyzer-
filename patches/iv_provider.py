"""
IV Provider — iv_provider.py
==============================
Fetches real ATM implied volatility from yfinance options chains
for an entire universe of tickers.

Targets DTE range 20-60 days for expiry selection.
Returns per-ticker: atm_iv, iv_percentile, iv_rank, selected_expiry, dte.
Falls back gracefully when options data is unavailable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("iv_provider")

TARGET_DTE_MIN = 20
TARGET_DTE_MAX = 60


def _select_expiry(expirations, target_min=TARGET_DTE_MIN, target_max=TARGET_DTE_MAX):
    """
    Select the best expiration date within the target DTE range.
    Prefers the expiry closest to 45 DTE (sweet spot for theta/vega balance).
    Returns (expiry_str, dte) or (None, None).
    """
    today = datetime.now().date()
    best = None
    best_dte = None
    best_dist = float('inf')
    ideal_dte = 45

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if target_min <= dte <= target_max:
            dist = abs(dte - ideal_dte)
            if dist < best_dist:
                best = exp_str
                best_dte = dte
                best_dist = dist

    # If nothing in range, try nearest expiry beyond 14 DTE
    if best is None:
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if dte >= 14:
                best = exp_str
                best_dte = dte
                break

    return best, best_dte


def _get_atm_iv(chain_calls, chain_puts, spot_price):
    """
    Extract ATM implied volatility from options chain.
    Finds the strike closest to spot price, averages call and put IV.
    Returns atm_iv as annualized percentage, or None.
    """
    if chain_calls.empty and chain_puts.empty:
        return None, None, None

    # Find ATM strike from calls
    call_iv = None
    put_iv = None
    atm_strike = None

    if not chain_calls.empty and 'strike' in chain_calls.columns:
        calls = chain_calls.dropna(subset=['impliedVolatility'])
        if not calls.empty:
            calls = calls[calls['impliedVolatility'] > 0.01]  # Filter garbage
            if not calls.empty:
                idx = (calls['strike'] - spot_price).abs().idxmin()
                atm_strike = calls.loc[idx, 'strike']
                call_iv = calls.loc[idx, 'impliedVolatility']

    if not chain_puts.empty and 'strike' in chain_puts.columns:
        puts = chain_puts.dropna(subset=['impliedVolatility'])
        if not puts.empty:
            puts = puts[puts['impliedVolatility'] > 0.01]
            if not puts.empty:
                if atm_strike is not None:
                    # Use same strike as calls
                    idx = (puts['strike'] - atm_strike).abs().idxmin()
                else:
                    idx = (puts['strike'] - spot_price).abs().idxmin()
                    atm_strike = puts.loc[idx, 'strike']
                put_iv = puts.loc[idx, 'impliedVolatility']

    # Average call and put IV (mid-market)
    ivs = [v for v in [call_iv, put_iv] if v is not None]
    if not ivs:
        return None, atm_strike, None

    atm_iv = np.mean(ivs)
    # yfinance returns IV as decimal (e.g., 0.35 = 35%)
    atm_iv_pct = round(atm_iv * 100, 2)

    return atm_iv_pct, atm_strike, round(atm_iv, 4)


def fetch_iv_single(symbol, spot_price=None):
    """
    Fetch ATM IV for a single ticker.

    Returns dict:
      {
        'atm_iv': float (annualized %),
        'atm_iv_raw': float (decimal),
        'atm_strike': float,
        'expiry': str,
        'dte': int,
        'call_volume': int,
        'put_volume': int,
        'source': 'options_chain' | 'unavailable'
      }
    """
    result = {
        'atm_iv': None,
        'atm_iv_raw': None,
        'atm_strike': None,
        'expiry': None,
        'dte': None,
        'call_volume': 0,
        'put_volume': 0,
        'source': 'unavailable',
    }

    try:
        t = yf.Ticker(symbol)

        # Get spot price if not provided
        if spot_price is None:
            hist = t.history(period='5d')
            if hist.empty:
                return result
            spot_price = float(hist['Close'].iloc[-1])

        # Get available expirations
        expirations = t.options
        if not expirations:
            return result

        # Select best expiry in target DTE range
        expiry, dte = _select_expiry(expirations)
        if expiry is None:
            return result

        # Fetch chain
        chain = t.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts

        # Extract ATM IV
        atm_iv_pct, atm_strike, atm_iv_raw = _get_atm_iv(calls, puts, spot_price)

        if atm_iv_pct is None:
            return result

        # Volume data
        call_vol = int(calls['volume'].sum()) if 'volume' in calls.columns else 0
        put_vol = int(puts['volume'].sum()) if 'volume' in puts.columns else 0
        if pd.isna(call_vol):
            call_vol = 0
        if pd.isna(put_vol):
            put_vol = 0

        result = {
            'atm_iv': atm_iv_pct,
            'atm_iv_raw': atm_iv_raw,
            'atm_strike': atm_strike,
            'expiry': expiry,
            'dte': dte,
            'call_volume': call_vol,
            'put_volume': put_vol,
            'source': 'options_chain',
        }

    except Exception as e:
        logger.debug(f"{symbol}: IV fetch failed - {e}")

    return result


def fetch_iv_universe(symbols, spot_prices=None):
    """
    Fetch real ATM IV for an entire universe of tickers.

    Args:
        symbols: list of ticker strings
        spot_prices: optional dict {symbol: float} of current prices

    Returns:
        dict: {symbol: {atm_iv, atm_iv_raw, atm_strike, expiry, dte,
                         call_volume, put_volume, source}}
    """
    if spot_prices is None:
        spot_prices = {}

    logger.info(f"Fetching real IV for {len(symbols)} tickers (target DTE {TARGET_DTE_MIN}-{TARGET_DTE_MAX})...")
    start = time.time()

    iv_data = {}
    success = 0
    failed = 0

    for i, sym in enumerate(symbols):
        spot = spot_prices.get(sym)
        data = fetch_iv_single(sym, spot)
        iv_data[sym] = data

        if data['source'] == 'options_chain':
            success += 1
        else:
            failed += 1

        # Progress every 10 tickers
        if (i + 1) % 10 == 0:
            logger.info(f"  IV progress: {i+1}/{len(symbols)} ({success} ok, {failed} failed)")

    elapsed = time.time() - start
    logger.info(
        f"IV fetch complete: {success}/{len(symbols)} with real IV, "
        f"{failed} will use HV proxy ({elapsed:.1f}s)"
    )

    return iv_data


def compute_iv_percentile_rank(current_iv, historical_ivs):
    """
    Compute IV percentile and rank from a series of historical ATM IVs.

    iv_percentile = % of observations below current
    iv_rank = (current - min) / (max - min) * 100
    """
    if not historical_ivs or current_iv is None:
        return None, None

    valid = [v for v in historical_ivs if v is not None and v > 0]
    if len(valid) < 5:
        return None, None

    below = sum(1 for v in valid if v < current_iv)
    iv_percentile = round(below / len(valid) * 100, 1)

    iv_min = min(valid)
    iv_max = max(valid)
    iv_range = iv_max - iv_min
    iv_rank = round((current_iv - iv_min) / iv_range * 100, 1) if iv_range > 0 else 50.0

    return iv_percentile, iv_rank


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    test_symbols = ['NVDA', 'AAPL', 'SMCI', 'PLTR', 'SPY']
    print(f"Testing IV fetch for {test_symbols}...")

    iv_data = fetch_iv_universe(test_symbols)

    for sym, data in iv_data.items():
        if data['source'] == 'options_chain':
            print(f"  {sym:6s}: IV={data['atm_iv']:.1f}%  strike={data['atm_strike']}  "
                  f"expiry={data['expiry']}  DTE={data['dte']}  "
                  f"C_vol={data['call_volume']:,}  P_vol={data['put_volume']:,}")
        else:
            print(f"  {sym:6s}: no options data (will use HV proxy)")
