"""
Patch 7: Stage-Enriched Scanner
Adds: stage_enriched_scan() and --mode stage_enriched to CLI

Runs unified_event_scan, then classifies all tickers using Weinstein Stage
Analysis with SPY as benchmark. Merges stages into output and applies
stage-adjusted scoring:
  +2.0 for sell signals in Stage 3/4
  +1.5 for buy signals in Stage 2 with CANSLIM stacking
  -1.0 for buy signals in Stage 4 (counter-trend penalty)
  +0.5 for sell signals in Stage 3 (topping confirmation)

Prints market regime header and stage distribution before results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging

from patches.options_overlay import apply_options_overlay
from patches.iv_provider import fetch_iv_universe
from patches.add_absolute_strength import apply_absolute_strength
from patches.add_comparative_strength import apply_comparative_strength

logger = logging.getLogger("stage_enriched")


def classify_universe(df, benchmark_df=None, symbol_col='symbol', date_col='date',
                      open_col='open', high_col='high', low_col='low',
                      close_col='close', volume_col='volume'):
    """
    Classify all tickers in the DataFrame into Weinstein Stages.
    Uses price vs 30-week SMA, SMA slope, 10-week momentum, volume,
    and 52-week position.

    Args:
        df: OHLCV DataFrame (multi-symbol)
        benchmark_df: DataFrame with benchmark (SPY) OHLCV for RS computation.
                      If None, RS is skipped.

    Returns:
        dict: {symbol: {stage, stage_label, confidence, mansfield_rs, details}}
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])

    # Compute benchmark 30-week SMA for Mansfield RS
    bench_sma_30w = None
    bench_close = None
    if benchmark_df is not None and not benchmark_df.empty:
        b = benchmark_df.sort_values(date_col).copy()
        bench_close = b[close_col].values
        bench_sma_30w = pd.Series(bench_close).rolling(150).mean().values

    results = {}

    for sym, g in df.groupby(symbol_col):
        g = g.copy().reset_index(drop=True)
        close = g[close_col].values
        vol = g[volume_col].values
        n = len(close)

        if n < 50:
            results[sym] = {
                'stage': None, 'stage_label': 'Insufficient Data',
                'confidence': 0, 'mansfield_rs': None, 'details': 'Need 50+ bars'
            }
            continue

        last = float(close[-1])

        # 30-week SMA (150 trading days)
        sma_30w = pd.Series(close).rolling(min(150, n)).mean()
        sma_30w_now = float(sma_30w.iloc[-1])
        sma_30w_prev = float(sma_30w.iloc[-20]) if n > 20 else sma_30w_now

        # 10-week SMA (50 trading days)
        sma_10w = pd.Series(close).rolling(min(50, n)).mean()
        sma_10w_now = float(sma_10w.iloc[-1])
        sma_10w_prev = float(sma_10w.iloc[-20]) if n > 20 else sma_10w_now

        # 52-week range
        tail_252 = close[-min(252, n):]
        high_52w = float(max(tail_252))
        low_52w = float(min(tail_252))

        # Volume
        vol_avg_50 = float(pd.Series(vol).rolling(50).mean().iloc[-1]) if n >= 50 else float(np.mean(vol))
        vol_recent = float(np.mean(vol[-5:])) if n >= 5 else vol_avg_50

        # ── Price vs 30w SMA ──
        pct_above_30w = ((last - sma_30w_now) / sma_30w_now) * 100 if sma_30w_now > 0 else 0

        # ── 30w SMA slope ──
        slope_30w = ((sma_30w_now - sma_30w_prev) / sma_30w_prev) * 100 if sma_30w_prev > 0 else 0

        # ── 10w momentum ──
        mom_10w = ((sma_10w_now - sma_10w_prev) / sma_10w_prev) * 100 if sma_10w_prev > 0 else 0

        # ── 52-week position ──
        position_52w = (last - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5

        # ── Volume ratio ──
        vol_ratio = vol_recent / vol_avg_50 if vol_avg_50 > 0 else 1.0

        # ── Mansfield RS ──
        mansfield_rs = None
        if bench_close is not None and len(bench_close) >= 150:
            # RS line = stock / benchmark
            min_len = min(n, len(bench_close))
            if min_len >= 150:
                stock_tail = close[-min_len:]
                bench_tail = bench_close[-min_len:]
                rs_line = stock_tail / bench_tail
                rs_sma = pd.Series(rs_line).rolling(150).mean().iloc[-1]
                if rs_sma > 0:
                    mansfield_rs = round(((rs_line[-1] / rs_sma) - 1) * 100, 2)

        # ── Stage Classification (weighted scoring) ──
        stage_scores = {1: 0, 2: 0, 3: 0, 4: 0}

        # Price vs 30w (weight: 0.25)
        if pct_above_30w > 5:
            stage_scores[2] += 0.25
        elif pct_above_30w > -2:
            stage_scores[3] += 0.15
            stage_scores[1] += 0.10
        elif pct_above_30w > -5:
            stage_scores[1] += 0.25
        else:
            stage_scores[4] += 0.25

        # 30w slope (weight: 0.20)
        if slope_30w > 0.5:
            stage_scores[2] += 0.20
        elif slope_30w > -0.2:
            stage_scores[3] += 0.20
        else:
            stage_scores[4] += 0.20

        # 10w momentum (weight: 0.15)
        if mom_10w > 1.0:
            stage_scores[2] += 0.15
        elif mom_10w > -0.5:
            stage_scores[3] += 0.10
            stage_scores[1] += 0.05
        else:
            stage_scores[4] += 0.15

        # 52-week position (weight: 0.15)
        if position_52w > 0.85:
            stage_scores[2] += 0.15
        elif position_52w > 0.5:
            stage_scores[3] += 0.10
            stage_scores[2] += 0.05
        elif position_52w > 0.2:
            stage_scores[1] += 0.15
        else:
            stage_scores[4] += 0.15

        # Volume pattern (weight: 0.10)
        if vol_ratio > 1.5:
            # Volume surge — could be breakout or breakdown
            stage_scores[2] += 0.05
            stage_scores[4] += 0.05
        elif vol_ratio < 0.7:
            stage_scores[1] += 0.10  # Dry-up = base building

        # Mansfield RS (weight: 0.15)
        if mansfield_rs is not None:
            if mansfield_rs > 1.0:
                stage_scores[2] += 0.15
            elif mansfield_rs > -1.0:
                stage_scores[3] += 0.10
            else:
                stage_scores[4] += 0.15

        # Winner
        stage = max(stage_scores, key=stage_scores.get)
        total = sum(stage_scores.values())
        confidence = round(stage_scores[stage] / total, 3) if total > 0 else 0

        stage_labels = {
            1: 'Stage 1 - Basing',
            2: 'Stage 2 - Advancing',
            3: 'Stage 3 - Topping',
            4: 'Stage 4 - Declining',
        }

        results[sym] = {
            'stage': stage,
            'stage_label': stage_labels[stage],
            'confidence': confidence,
            'mansfield_rs': mansfield_rs,
            'pct_above_30w': round(pct_above_30w, 2),
            'slope_30w': round(slope_30w, 2),
            'momentum_10w': round(mom_10w, 2),
            'position_52w': round(position_52w, 3),
            'vol_ratio': round(vol_ratio, 2),
        }

    return results


def get_market_regime(benchmark_df, close_col='close'):
    """
    Determine market regime from benchmark (SPY) data.
    Returns dict with regime classification and supporting data.
    """
    if benchmark_df is None or benchmark_df.empty:
        return {'regime': 'Unknown', 'details': 'No benchmark data'}

    close = benchmark_df[close_col].values
    n = len(close)
    if n < 200:
        return {'regime': 'Unknown', 'details': 'Insufficient benchmark data'}

    last = float(close[-1])
    sma_50 = float(pd.Series(close).rolling(50).mean().iloc[-1])
    sma_200 = float(pd.Series(close).rolling(200).mean().iloc[-1])
    sma_50_prev = float(pd.Series(close).rolling(50).mean().iloc[-20])

    pct_above_200 = ((last - sma_200) / sma_200) * 100
    slope_50 = ((sma_50 - sma_50_prev) / sma_50_prev) * 100

    # 20-day realized volatility
    returns = pd.Series(close).pct_change().dropna()
    vol_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)

    # A/D proxy: % of last 20 days that were up
    up_pct = float((returns.tail(20) > 0).mean() * 100)

    # Regime classification
    if last > sma_50 > sma_200 and slope_50 > 0.3:
        regime = 'RISK-ON (Confirmed Uptrend)'
    elif last > sma_200 and slope_50 > 0:
        regime = 'RISK-ON (Moderate)'
    elif last > sma_200 and slope_50 <= 0:
        regime = 'TRANSITIONAL (Weakening)'
    elif last < sma_200 and last > sma_50:
        regime = 'TRANSITIONAL (Recovery Attempt)'
    elif last < sma_50 < sma_200:
        regime = 'RISK-OFF (Confirmed Downtrend)'
    elif last < sma_200:
        regime = 'RISK-OFF (Below 200d)'
    else:
        regime = 'NEUTRAL'

    return {
        'regime': regime,
        'spy_price': round(last, 2),
        'spy_vs_200sma': round(pct_above_200, 2),
        'spy_50sma_slope': round(slope_50, 2),
        'volatility_20d': round(vol_20d, 1),
        'up_day_pct_20d': round(up_pct, 1),
    }


def stage_enriched_scan(df, symbol_col='symbol', date_col='date',
                        open_col='open', high_col='high', low_col='low',
                        close_col='close', volume_col='volume',
                        benchmark_symbol='SPY'):
    """
    Run unified event scan, classify all tickers into Weinstein Stages,
    merge stages, and apply stage-adjusted scoring.

    Stage-adjusted scoring:
      +2.0  sell signals in Stage 3/4
      +1.5  buy signals in Stage 2 with CANSLIM stacking (breakaway + vol + RS)
      +0.5  sell signals in Stage 3 (topping confirmation)
      -1.0  buy signals in Stage 4 (counter-trend penalty)
      +0.5  any signal aligned with stage direction
    """
    from patches.add_unified_scanner import unified_event_scan

    # ── Extract benchmark data ──
    benchmark_df = None
    if benchmark_symbol in df[symbol_col].values:
        benchmark_df = df[df[symbol_col] == benchmark_symbol].copy().reset_index(drop=True)
    else:
        # Try downloading SPY
        try:
            import yfinance as yf
            spy = yf.Ticker(benchmark_symbol)
            bdf = spy.history(period='1y', interval='1d').reset_index()
            bdf = bdf.rename(columns={
                'Date': date_col, 'Open': open_col, 'High': high_col,
                'Low': low_col, 'Close': close_col, 'Volume': volume_col
            })
            bdf[date_col] = pd.to_datetime(bdf[date_col]).dt.tz_localize(None)
            bdf[symbol_col] = benchmark_symbol
            benchmark_df = bdf
            # Add to main df for scanning
            df = pd.concat([df, bdf[[symbol_col, date_col, open_col, high_col,
                                     low_col, close_col, volume_col]]], ignore_index=True)
            print(f"Downloaded {benchmark_symbol} as benchmark ({len(bdf)} bars)")
        except Exception as e:
            print(f"Could not fetch {benchmark_symbol}: {e}")

    # ── Market Regime ──
    regime = get_market_regime(benchmark_df, close_col)

    print()
    print("=" * 60)
    print("  MARKET REGIME")
    print("=" * 60)
    print(f"  Regime:          {regime.get('regime', 'Unknown')}")
    if 'spy_price' in regime:
        print(f"  SPY:             ${regime['spy_price']}")
        print(f"  SPY vs 200 SMA:  {regime['spy_vs_200sma']:+.2f}%")
        print(f"  50 SMA Slope:    {regime['spy_50sma_slope']:+.2f}%")
        print(f"  Volatility 20d:  {regime['volatility_20d']:.1f}%")
        print(f"  Up Days (20d):   {regime['up_day_pct_20d']:.0f}%")
    print("=" * 60)
    print()

    # ── Run unified event scan ──
    result = unified_event_scan(
        df, symbol_col, date_col, open_col,
        high_col, low_col, close_col, volume_col
    )

    # Remove benchmark from results
    result = result[result[symbol_col] != benchmark_symbol].copy()

    # ── Classify universe into stages ──
    print("\nClassifying Weinstein Stages...")
    stages = classify_universe(
        df, benchmark_df, symbol_col, date_col,
        open_col, high_col, low_col, close_col, volume_col
    )

    # Print stage distribution
    stage_counts = {}
    for sym, info in stages.items():
        if sym == benchmark_symbol:
            continue
        label = info.get('stage_label', 'Unknown')
        stage_counts[label] = stage_counts.get(label, 0) + 1

    print()
    print("=" * 60)
    print("  STAGE DISTRIBUTION")
    print("=" * 60)
    for label, count in sorted(stage_counts.items()):
        bar = '#' * (count * 3)
        print(f"  {label:30s} {count:3d}  {bar}")
    print("=" * 60)
    print()

    # ── Merge stages into result ──
    stage_df = pd.DataFrame([
        {'symbol': sym, 'w_stage': info['stage'], 'w_stage_label': info['stage_label'],
         'w_confidence': info['confidence'], 'w_mansfield_rs': info.get('mansfield_rs'),
         'w_pct_above_30w': info.get('pct_above_30w'),
         'w_slope_30w': info.get('slope_30w')}
        for sym, info in stages.items()
        if info.get('stage') is not None
    ])

    if not stage_df.empty:
        result = result.merge(stage_df, on='symbol', how='left')
    else:
        result['w_stage'] = None
        result['w_stage_label'] = None
        result['w_confidence'] = None
        result['w_mansfield_rs'] = None

    # ── Stage-Adjusted Scoring ──
    print("Applying stage-adjusted scoring...")
    result['stage_adj'] = 0.0
    result['stage_adj_reason'] = ''

    for idx in result.index:
        stage = result.loc[idx, 'w_stage']
        event_code = result.loc[idx, 'event_code'] if 'event_code' in result.columns else 'none'
        direction = result.loc[idx, 'direction'] if 'direction' in result.columns else ''
        sell = result.loc[idx, 'sell_signal'] if 'sell_signal' in result.columns else False
        is_bullish = direction == 'bullish' or 'bull' in str(event_code)
        is_bearish = direction == 'bearish' or 'bear' in str(event_code) or sell

        if pd.isna(stage) or event_code == 'none':
            continue

        adj = 0.0
        reasons = []

        # +2.0: sell signals in Stage 3/4
        if is_bearish and stage in (3, 4) and sell:
            adj += 2.0
            reasons.append(f'Sell in Stage {stage} (+2.0)')

        # +0.5: sell signals in Stage 3 (topping confirmation)
        elif is_bearish and stage == 3:
            adj += 0.5
            reasons.append('Bearish in Stage 3 (+0.5)')

        # +1.5: buy signals in Stage 2 with CANSLIM stacking
        # CANSLIM stack = breakaway gap + volume spike + positive RS
        if is_bullish and stage == 2:
            is_breakaway = 'breakaway' in str(event_code)
            has_vol = result.loc[idx, 'vol_spike'] if 'vol_spike' in result.columns else False
            has_rs = (result.loc[idx, 'w_mansfield_rs'] or 0) > 0
            canslim_stack = is_breakaway and has_vol and has_rs

            if canslim_stack:
                adj += 1.5
                reasons.append('Stage 2 + CANSLIM stack (+1.5)')
            else:
                adj += 0.5
                reasons.append('Bullish in Stage 2 (+0.5)')

        # -1.0: buy signals in Stage 4 (counter-trend penalty)
        if is_bullish and stage == 4:
            adj -= 1.0
            reasons.append('Counter-trend buy in Stage 4 (-1.0)')

        # +0.5: signal aligned with stage direction
        if (is_bullish and stage == 2) or (is_bearish and stage == 4):
            adj += 0.5
            reasons.append('Stage-aligned (+0.5)')

        result.loc[idx, 'stage_adj'] = round(adj, 1)
        result.loc[idx, 'stage_adj_reason'] = '; '.join(reasons)

    # Apply adjustment to event_score
    if 'event_score' in result.columns:
        result['event_score_raw'] = result['event_score']
        result['event_score'] = (result['event_score'] + result['stage_adj']).clip(0, 12).round(1)

    # ── Absolute Strength ──
    print("Computing absolute strength metrics...")
    result = apply_absolute_strength(result, df, symbol_col, date_col, close_col)

    # ── Comparative Strength (sector rotation + intra-sector rank) ──
    print("Computing comparative strength (sector rotation, intra-sector rankings)...")
    result = apply_comparative_strength(result, df, symbol_col, date_col, close_col)

    # ── Fetch Real IV ──
    scan_symbols = [s for s in result[symbol_col].unique() if s != 'SPY']
    # Build spot prices from latest close in OHLCV data
    spot_prices = {}
    for sym, g in df.sort_values(date_col).groupby(symbol_col):
        spot_prices[sym] = float(g[close_col].iloc[-1])

    print("Fetching real ATM IV from options chains...")
    iv_data = fetch_iv_universe(scan_symbols, spot_prices=spot_prices)

    real_count = sum(1 for v in iv_data.values() if v.get('source') == 'options_chain')
    proxy_count = len(iv_data) - real_count
    print(f"  Real IV: {real_count} tickers | HV proxy fallback: {proxy_count} tickers")

    # ── Options Overlay (with real IV where available) ──
    print("Applying options overlay (real IV + HV fallback)...")
    result = apply_options_overlay(result, df, symbol_col, date_col, close_col, iv_data=iv_data)

    # Print overlay summary
    if 'iv_source' in result.columns:
        src_dist = result.drop_duplicates(subset=[symbol_col])['iv_source'].value_counts()
        print(f"\n  IV Source breakdown:")
        for src, count in src_dist.items():
            print(f"    {src:20s}: {count}")

    if 'iv_regime' in result.columns:
        iv_dist = result.drop_duplicates(subset=[symbol_col])['iv_regime'].value_counts()
        print(f"\n  IV Regime distribution:")
        for regime_name, count in iv_dist.items():
            print(f"    {regime_name:8s}: {count}")

    if 'structure' in result.columns:
        struct_dist = result.drop_duplicates(subset=[symbol_col])['structure'].value_counts()
        print(f"\n  Recommended structures:")
        for struct, count in struct_dist.items():
            print(f"    {struct}")

    if 'hard_rule_flags' in result.columns:
        violations = result[result['hard_rule_flags'].fillna('') != '']
        if len(violations) > 0:
            print(f"\n  !! HARD RULE VIOLATIONS: {len(violations)} rows")
            for _, row in violations.drop_duplicates(subset=[symbol_col]).iterrows():
                print(f"    {row[symbol_col]}: {row['hard_rule_flags']}")

    # Re-sort by adjusted score
    result = result.sort_values(['event_score', 'date'], ascending=[False, False])

    # Summary
    adjusted_count = (result['stage_adj'] != 0).sum()
    print(f"\nStage adjustments applied: {adjusted_count} rows modified")

    return result, regime, stages


if __name__ == '__main__':
    from breakaway_gap_scan import breakaway_gap_scan

    print("Patch 7: Running stage-enriched scan...")
    df = pd.read_csv('ohlcv.csv')
    result, regime, stages = stage_enriched_scan(df)

    events = result[result.get('event_code', pd.Series('none')) != 'none']
    print(f"\nTop 20 stage-enriched events:")
    display_cols = [c for c in [
        'symbol', 'date', 'close', 'event_label', 'event_score',
        'event_score_raw', 'stage_adj', 'w_stage_label', 'stage_adj_reason'
    ] if c in result.columns]
    print(events[display_cols].head(20).to_string(index=False))

    result.to_csv('stage_enriched_results.csv', index=False)
    print("\nSaved to stage_enriched_results.csv")
