"""
Options Overlay — options_overlay.py
=====================================
Computes volatility metrics and recommends options structures
based on Weinstein Stage x IV Regime mapping.

Metrics per ticker:
  hv_20, hv_60          — 20/60-day historical (realized) volatility
  iv_percentile         — IV percentile (52-week rank), uses HV as proxy if no chain
  iv_rank               — IV rank (current vs 52w range)
  hv_iv_ratio           — HV/IV ratio (>1 = IV cheap relative to realized)
  iv_regime             — 'cheap', 'fair', 'rich'
  hv_iv_signal          — directional signal from HV/IV divergence
  structure             — recommended options play (stage x IV regime)
  dte_range             — recommended days to expiration
  strike_guidance       — strike selection guidance
  hard_rule_flags       — list of hard rule violations if any

Stage x IV Regime mapping:
  Stage 2A + cheap IV  -> Long calls, 45-90 DTE
  Stage 2A + rich IV   -> Bull call spread, 45-90 DTE
  Stage 2B             -> Collar or covered calls (defensive)
  Stage 3 + cheap IV   -> Long puts, 30-60 DTE
  Stage 3 + rich IV    -> Bear put spread, 30-60 DTE
  Stage 4              -> Bear put spread, 30-60 DTE

Hard rules:
  NEVER sell ATM options in Stage 3/4
  NEVER sell naked puts in Stage 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("options_overlay")


def compute_hv(close_prices, window=20):
    """Annualized historical volatility from close prices."""
    if len(close_prices) < window + 1:
        return None
    returns = np.log(close_prices / np.roll(close_prices, 1))[1:]
    if len(returns) < window:
        return None
    hv = float(np.std(returns[-window:]) * np.sqrt(252) * 100)
    return round(hv, 2)


def compute_iv_proxy(close_prices, short_window=20, long_window=252):
    """
    IV proxy when no options chain is available.
    Uses the 52-week HV range to estimate IV percentile and rank.
    The current short-window HV is ranked against rolling HV over 1 year.
    """
    n = len(close_prices)
    if n < long_window + 1:
        # Fall back to what we have
        long_window = max(n - 1, short_window + 10)
        if long_window <= short_window:
            return None, None, None

    returns = np.log(close_prices / np.roll(close_prices, 1))[1:]

    # Rolling HV series over the long window
    rolling_hv = []
    for i in range(short_window, len(returns)):
        window_returns = returns[i - short_window:i]
        hv = float(np.std(window_returns) * np.sqrt(252) * 100)
        rolling_hv.append(hv)

    if len(rolling_hv) < 10:
        return None, None, None

    current_hv = rolling_hv[-1]
    hv_min = min(rolling_hv)
    hv_max = max(rolling_hv)
    hv_range = hv_max - hv_min

    # IV Rank = (current - min) / (max - min)
    iv_rank = round((current_hv - hv_min) / hv_range * 100, 1) if hv_range > 0 else 50.0

    # IV Percentile = % of observations below current
    below = sum(1 for h in rolling_hv if h < current_hv)
    iv_percentile = round(below / len(rolling_hv) * 100, 1)

    return iv_percentile, iv_rank, current_hv


def classify_iv_regime(iv_percentile):
    """Classify IV into cheap/fair/rich."""
    if iv_percentile is None:
        return 'unknown'
    if iv_percentile <= 25:
        return 'cheap'
    elif iv_percentile <= 65:
        return 'fair'
    else:
        return 'rich'


def hv_iv_signal(hv_20, hv_60, iv_proxy):
    """
    Signal from HV/IV divergence.
    - HV20 > IV and rising: vol expansion, momentum move likely continuing
    - HV20 < IV and falling: vol contraction, mean reversion likely
    """
    if hv_20 is None or hv_60 is None or iv_proxy is None:
        return 'neutral', 1.0

    ratio = hv_20 / iv_proxy if iv_proxy > 0 else 1.0

    if hv_20 > iv_proxy * 1.1 and hv_20 > hv_60:
        return 'vol_expanding', round(ratio, 2)
    elif hv_20 < iv_proxy * 0.85 and hv_20 < hv_60:
        return 'vol_contracting', round(ratio, 2)
    elif hv_20 > hv_60 * 1.15:
        return 'hv_spike', round(ratio, 2)
    else:
        return 'neutral', round(ratio, 2)


def classify_stage_substage(stage, pct_above_30w=None, slope_30w=None, momentum_10w=None):
    """
    Subdivide Stage 2 into 2A (early/strong advance) and 2B (late/defensive).
    Stage 2A: strong slope, good momentum, well above 30w
    Stage 2B: flattening slope or weakening momentum
    """
    if stage != 2:
        return str(stage)

    if pct_above_30w is not None and slope_30w is not None:
        if slope_30w > 0.3 and (pct_above_30w or 0) > 5:
            return '2A'
        else:
            return '2B'

    # Default to 2A if no detail data
    return '2A'


def recommend_structure(stage_sub, iv_regime, direction=None):
    """
    Map stage x IV regime to recommended options structure.

    Returns: (structure, dte_range, strike_guidance)
    """
    # ── Stage 2A: Advancing, early/strong ──
    if stage_sub == '2A':
        if iv_regime == 'cheap':
            return (
                'Long Calls',
                '45-90 DTE',
                'ATM to slightly OTM (delta 0.50-0.40). Cheap IV = favorable long premium.'
            )
        elif iv_regime == 'rich':
            return (
                'Bull Call Spread',
                '45-90 DTE',
                'Buy ATM call, sell OTM call 5-10% above. Caps cost in high-IV environment.'
            )
        else:  # fair
            return (
                'Long Calls or Bull Call Spread',
                '45-90 DTE',
                'ATM to slight OTM. Consider spread if IV rank > 40.'
            )

    # ── Stage 2B: Advancing but late/defensive ──
    if stage_sub == '2B':
        return (
            'Collar or Covered Calls',
            '30-60 DTE',
            'Defensive posture. Sell OTM calls (delta 0.30) to fund OTM put protection (delta -0.25). '
            'Stage 2B = momentum weakening, protect gains.'
        )

    # ── Stage 3: Topping ──
    if stage_sub == '3':
        if iv_regime == 'cheap':
            return (
                'Long Puts',
                '30-60 DTE',
                'ATM to slightly ITM puts (delta -0.55 to -0.45). '
                'Cheap IV = favorable long premium for downside.'
            )
        elif iv_regime == 'rich':
            return (
                'Bear Put Spread',
                '30-60 DTE',
                'Buy ATM put, sell OTM put 5-10% below. Rich IV makes outright puts expensive.'
            )
        else:  # fair
            return (
                'Long Puts or Bear Put Spread',
                '30-60 DTE',
                'Stage 3 topping. Favor puts; use spread if IV rank > 50.'
            )

    # ── Stage 4: Declining ──
    if stage_sub == '4':
        return (
            'Bear Put Spread',
            '30-60 DTE',
            'Buy ATM put, sell OTM put 7-15% below. Do NOT sell naked puts. '
            'Stage 4 can accelerate. Spread defines max risk.'
        )

    # ── Stage 1: Basing ──
    if stage_sub == '1':
        return (
            'No Position or Small Long Calls on Breakout',
            '60-120 DTE',
            'Wait for Stage 2 confirmation. If entering early, use small long calls '
            'with wide DTE to survive the base-building phase.'
        )

    return ('No Recommendation', 'N/A', 'Insufficient stage data')


def check_hard_rules(stage_sub, structure):
    """
    Enforce hard rules. Returns list of violation flags.
    NEVER sell ATM options in Stage 3/4
    NEVER sell naked puts in Stage 4
    """
    flags = []
    structure_lower = structure.lower()

    # Check: no ATM selling in Stage 3/4
    if stage_sub in ('3', '4'):
        atm_sells = ['short straddle', 'short strangle', 'naked call', 'sell atm',
                      'iron butterfly', 'short atm']
        for pattern in atm_sells:
            if pattern in structure_lower:
                flags.append(f'HARD RULE: No ATM selling in Stage {stage_sub}')
                break

    # Check: no naked puts in Stage 4
    if stage_sub == '4':
        naked_put_patterns = ['naked put', 'short put', 'sell put', 'cash secured put']
        for pattern in naked_put_patterns:
            if pattern in structure_lower:
                flags.append('HARD RULE: No naked/short puts in Stage 4')
                break

    return flags


def options_overlay_for_ticker(close_prices, stage=None, pct_above_30w=None,
                                slope_30w=None, momentum_10w=None, direction=None,
                                real_iv=None):
    """
    Compute all options overlay metrics for a single ticker.

    Args:
        close_prices: array/list of daily close prices (oldest first)
        stage: Weinstein stage (1-4) or None
        pct_above_30w: % price above 30w SMA
        slope_30w: 30w SMA slope
        momentum_10w: 10w momentum
        direction: 'bullish' or 'bearish' from event scan
        real_iv: dict from iv_provider.fetch_iv_single() with real ATM IV data.
                 If provided and source='options_chain', uses real IV instead of HV proxy.

    Returns:
        dict with all overlay fields
    """
    close = np.array(close_prices, dtype=float)

    # ── HV metrics (always computed) ──
    hv_20 = compute_hv(close, 20)
    hv_60 = compute_hv(close, 60)

    # ── IV: use real IV if available, fall back to HV proxy ──
    iv_source = 'hv_proxy'
    atm_iv = None
    atm_strike = None
    iv_expiry = None
    iv_dte = None

    if real_iv and real_iv.get('source') == 'options_chain' and real_iv.get('atm_iv'):
        # Real IV from options chain
        atm_iv = real_iv['atm_iv']
        atm_strike = real_iv.get('atm_strike')
        iv_expiry = real_iv.get('expiry')
        iv_dte = real_iv.get('dte')
        iv_source = 'options_chain'

        # Compute IV percentile/rank using HV history as reference distribution
        # (how does current real IV compare to historical HV range?)
        hv_pctile, hv_rank, _ = compute_iv_proxy(close)
        # Adjust: if real IV is higher than HV proxy, percentile should be higher
        if hv_pctile is not None and _ is not None and _ > 0:
            iv_ratio_vs_hv = atm_iv / _
            # Scale the HV-based percentile by how much real IV exceeds/undercuts HV
            iv_percentile = min(99.0, max(1.0, round(hv_pctile * iv_ratio_vs_hv, 1)))
            iv_rank = min(99.0, max(1.0, round((hv_rank or 50) * iv_ratio_vs_hv, 1)))
        else:
            iv_percentile = hv_pctile
            iv_rank = hv_rank

        iv_for_ratio = atm_iv
    else:
        # Fall back to HV proxy
        iv_percentile, iv_rank, iv_for_ratio = compute_iv_proxy(close)
        atm_iv = iv_for_ratio  # HV proxy value

    iv_regime = classify_iv_regime(iv_percentile)
    hv_iv_sig, hv_iv_ratio = hv_iv_signal(hv_20, hv_60, iv_for_ratio)

    # ── Stage substage ──
    stage_sub = classify_stage_substage(stage, pct_above_30w, slope_30w, momentum_10w)

    # ── Structure recommendation ──
    structure, dte_range, strike_guidance = recommend_structure(stage_sub, iv_regime, direction)

    # ── Hard rules ──
    hard_flags = check_hard_rules(stage_sub, structure)

    return {
        'hv_20': hv_20,
        'hv_60': hv_60,
        'atm_iv': atm_iv,
        'iv_percentile': iv_percentile,
        'iv_rank': iv_rank,
        'hv_iv_ratio': hv_iv_ratio,
        'iv_regime': iv_regime,
        'iv_source': iv_source,
        'iv_expiry': iv_expiry,
        'iv_dte': iv_dte,
        'atm_strike': atm_strike,
        'hv_iv_signal': hv_iv_sig,
        'stage_substage': stage_sub,
        'structure': structure,
        'dte_range': dte_range,
        'strike_guidance': strike_guidance,
        'hard_rule_flags': '; '.join(hard_flags) if hard_flags else '',
    }


def apply_options_overlay(df, ohlcv_df, symbol_col='symbol', date_col='date',
                          close_col='close', iv_data=None):
    """
    Apply options overlay to an enriched scan DataFrame.
    Uses the raw OHLCV data to compute per-ticker volatility metrics,
    then maps structure recommendations based on stage + IV regime.

    Args:
        df: enriched scan result (must have w_stage, w_pct_above_30w, w_slope_30w)
        ohlcv_df: raw OHLCV DataFrame for close price arrays
        iv_data: optional dict from iv_provider.fetch_iv_universe().
                 {symbol: {atm_iv, source, ...}}. If provided, uses real IV
                 where available, falls back to HV proxy per ticker.

    Returns:
        df with overlay columns added
    """
    if iv_data is None:
        iv_data = {}

    df = df.copy()
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df[date_col] = pd.to_datetime(ohlcv_df[date_col])
    ohlcv_df = ohlcv_df.sort_values([symbol_col, date_col])

    # Build close price arrays per symbol
    close_arrays = {}
    for sym, g in ohlcv_df.groupby(symbol_col):
        close_arrays[sym] = g[close_col].values

    # Initialize overlay columns
    overlay_cols = [
        'hv_20', 'hv_60', 'atm_iv', 'iv_percentile', 'iv_rank', 'hv_iv_ratio',
        'iv_regime', 'iv_source', 'iv_expiry', 'iv_dte', 'atm_strike',
        'hv_iv_signal', 'stage_substage', 'structure',
        'dte_range', 'strike_guidance', 'hard_rule_flags'
    ]
    for col in overlay_cols:
        df[col] = None

    # Cache per-symbol overlays (same for all rows of a symbol)
    overlay_cache = {}

    real_iv_count = 0
    hv_proxy_count = 0

    symbols = df[symbol_col].unique()
    for sym in symbols:
        if sym in overlay_cache:
            continue

        close_arr = close_arrays.get(sym)
        if close_arr is None or len(close_arr) < 30:
            overlay_cache[sym] = {col: None for col in overlay_cols}
            continue

        # Get stage info from first row of this symbol
        sym_rows = df[df[symbol_col] == sym]
        stage = sym_rows['w_stage'].iloc[0] if 'w_stage' in df.columns else None
        pct_above = sym_rows['w_pct_above_30w'].iloc[0] if 'w_pct_above_30w' in df.columns else None
        slope = sym_rows['w_slope_30w'].iloc[0] if 'w_slope_30w' in df.columns else None
        direction = sym_rows['direction'].iloc[0] if 'direction' in df.columns else None

        # Handle NaN
        if pd.isna(stage):
            stage = None
        if pd.isna(pct_above):
            pct_above = None
        if pd.isna(slope):
            slope = None

        # Get real IV for this ticker (if available)
        ticker_iv = iv_data.get(sym)

        overlay = options_overlay_for_ticker(
            close_arr, stage=stage, pct_above_30w=pct_above,
            slope_30w=slope, direction=direction, real_iv=ticker_iv
        )
        overlay_cache[sym] = overlay

        if overlay.get('iv_source') == 'options_chain':
            real_iv_count += 1
        else:
            hv_proxy_count += 1

    logger.info(f"IV sources: {real_iv_count} real options chain, {hv_proxy_count} HV proxy")

    # Apply to DataFrame
    for sym, overlay in overlay_cache.items():
        mask = df[symbol_col] == sym
        for col, val in overlay.items():
            df.loc[mask, col] = val

    logger.info(f"Options overlay applied to {len(overlay_cache)} symbols")
    return df


if __name__ == '__main__':
    # Quick test with sample data
    import yfinance as yf

    sym = 'NVDA'
    print(f"Testing options overlay for {sym}...")
    t = yf.Ticker(sym)
    hist = t.history(period='1y', interval='1d')
    close = hist['Close'].values

    result = options_overlay_for_ticker(close, stage=3, pct_above_30w=-1.5, slope_30w=-0.3)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
