"""
Patch 15: Master Score — Unified Signal Ranking
=================================================
Combines all pipeline layers into a single master_score (0-100)
that ranks every signal by total conviction.

Pipeline flow:
  unified events → tradeability score → sector rank → theme crossovers → master_score

Components (weighted):
  30% — event_score (normalized 0-12 → 0-30)
         Base signal quality: gap type, stage adjustment, climax/trendline
  20% — tradeability_score (new, 0-20)
         Can you actually trade this? IV regime, options liquidity, spread width,
         fill risk, HV/IV ratio
  15% — abs_strength_score (normalized 0-10 → 0-15)
         Is this a leader or a dog? Universe rank, Mansfield RS, RS slope
  15% — sector_rank_score (normalized 0-15)
         Is the sector rotating in? Cross-sector rank, rotation signal, intra-sector position
  10% — theme_momentum_score (normalized 0-10)
         Is the theme accelerating? 20d rank change, trajectory
  10% — confluence_bonus (0-10)
         Multiple signals agreeing: stage+strength aligned, sector+theme aligned,
         event+IV regime aligned

Plus:
  theme_crossover detection — flags when a sector crosses above/below
  the 50th percentile theme rank (rotation inflection point)

CLI: --mode master
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from datetime import datetime

from patches.add_stage_enriched import stage_enriched_scan
from patches.add_theme_momentum import theme_momentum_timeseries
from patches.add_comparative_strength import _get_sector

logger = logging.getLogger("master_score")


# ══════════════════════════════════════════════════════════════
# 1. TRADEABILITY SCORE (0-20)
# ══════════════════════════════════════════════════════════════

def compute_tradeability(df):
    """
    Score how tradeable each signal is (0-20).

    Components:
      0-5  IV regime fit: cheap IV + long = good, rich IV + spread = good
      0-4  Options liquidity proxy: volume confirmation, not a microcap
      0-3  Fill risk: low fill risk = gap likely to hold
      0-3  HV/IV alignment: HV < IV = premium seller edge, HV > IV = buyer edge
      0-3  Structure clarity: hard rules clean, DTE in range
      0-2  Spread feasibility: not a $5 stock with wide spreads
    """
    df = df.copy()
    df['tradeability_score'] = 0.0

    for idx in df.index:
        score = 0.0

        # ── IV regime fit (0-5) ──
        iv_regime = df.loc[idx, 'iv_regime'] if 'iv_regime' in df.columns else None
        structure = str(df.loc[idx, 'structure']) if 'structure' in df.columns else ''
        direction = df.loc[idx, 'direction'] if 'direction' in df.columns else ''
        is_long = 'long' in structure.lower() and 'spread' not in structure.lower()
        is_spread = 'spread' in structure.lower()

        if iv_regime == 'cheap' and is_long:
            score += 5  # Best case: cheap IV + outright long
        elif iv_regime == 'rich' and is_spread:
            score += 4  # Rich IV correctly matched with spread
        elif iv_regime == 'fair':
            score += 3  # Neutral
        elif iv_regime == 'cheap' and is_spread:
            score += 2  # Cheap IV but using spread (suboptimal)
        elif iv_regime == 'rich' and is_long:
            score += 1  # Rich IV + outright long (paying up)

        # ── Options liquidity proxy (0-4) ──
        vol_conf = df.loc[idx, 'volume_confirmation'] if 'volume_confirmation' in df.columns else None
        vol_spike = df.loc[idx, 'vol_spike'] if 'vol_spike' in df.columns else None
        close_price = df.loc[idx, 'close'] if 'close' in df.columns else 0

        if vol_conf or vol_spike:
            score += 2
        if close_price and not pd.isna(close_price) and float(close_price) > 20:
            score += 1  # Not a penny stock
        if close_price and not pd.isna(close_price) and float(close_price) > 50:
            score += 1  # Liquid enough for decent spreads

        # ── Fill risk (0-3) ──
        fill_risk = df.loc[idx, 'gap_fill_risk'] if 'gap_fill_risk' in df.columns else 'n/a'
        if fill_risk == 'low':
            score += 3
        elif fill_risk == 'medium':
            score += 1.5
        elif fill_risk == 'n/a':
            score += 2  # Not a gap signal, neutral

        # ── HV/IV alignment (0-3) ──
        hv_iv_sig = df.loc[idx, 'hv_iv_signal'] if 'hv_iv_signal' in df.columns else 'neutral'
        is_bearish = direction == 'bearish' or (
            df.loc[idx, 'sell_signal'] if 'sell_signal' in df.columns else False
        )

        if hv_iv_sig == 'vol_expanding' and is_bearish:
            score += 3  # Vol expanding + bearish = puts work
        elif hv_iv_sig == 'vol_contracting' and not is_bearish:
            score += 2  # Vol contracting + bullish = calls cheap
        elif hv_iv_sig == 'neutral':
            score += 1.5

        # ── Structure clarity (0-3) ──
        hard_flags = str(df.loc[idx, 'hard_rule_flags']) if 'hard_rule_flags' in df.columns else ''
        if not hard_flags or hard_flags == '' or hard_flags == 'nan':
            score += 2  # Clean, no violations
        else:
            score -= 1  # Has violations

        dte = str(df.loc[idx, 'dte_range']) if 'dte_range' in df.columns else ''
        if '45-90' in dte or '30-60' in dte:
            score += 1  # Standard DTE range

        # ── Spread feasibility (0-2) ──
        if close_price and not pd.isna(close_price) and float(close_price) > 30:
            score += 1
        if close_price and not pd.isna(close_price) and float(close_price) > 100:
            score += 1  # Wide strike availability

        df.loc[idx, 'tradeability_score'] = round(min(score, 20.0), 1)

    # tradeability_flag: True if score >= 12 (tradeable with acceptable structure)
    df['tradeability_flag'] = df['tradeability_score'] >= 12

    return df


# ══════════════════════════════════════════════════════════════
# 2. THEME CROSSOVER DETECTION
# ══════════════════════════════════════════════════════════════

def detect_theme_crossovers(sector_ts, date_col='date', lookback=5):
    """
    Detect sectors crossing above/below the 50th percentile theme rank.

    A crossover UP = sector emerging as rotation target.
    A crossover DOWN = sector losing theme momentum.

    Returns dict: {sector: {direction, cross_date, rank_before, rank_after}}
    """
    crossovers = {}

    if sector_ts is None or sector_ts.empty:
        return crossovers

    latest_date = sector_ts[date_col].max()
    recent = sector_ts[sector_ts[date_col] >= latest_date - pd.Timedelta(days=lookback * 2)]

    for sector, g in recent.groupby('sector'):
        g = g.sort_values(date_col)
        if len(g) < 3:
            continue

        ranks = g['theme_rank'].values
        current = ranks[-1]

        # Look for 50th percentile crossing in last N points
        prev_below_50 = any(r < 50 for r in ranks[:-1])
        prev_above_50 = any(r >= 50 for r in ranks[:-1])
        earliest_rank = ranks[0]

        if current >= 50 and prev_below_50 and earliest_rank < 50:
            crossovers[sector] = {
                'direction': 'CROSS_UP',
                'rank_before': round(float(earliest_rank), 1),
                'rank_after': round(float(current), 1),
                'cross_date': str(latest_date)[:10],
            }
        elif current < 50 and prev_above_50 and earliest_rank >= 50:
            crossovers[sector] = {
                'direction': 'CROSS_DOWN',
                'rank_before': round(float(earliest_rank), 1),
                'rank_after': round(float(current), 1),
                'cross_date': str(latest_date)[:10],
            }

    return crossovers


# ══════════════════════════════════════════════════════════════
# 3. SECTOR RANK SCORE (0-15)
# ══════════════════════════════════════════════════════════════

def compute_sector_rank_score(df):
    """
    Convert sector ranking data into a 0-15 score.

    Components:
      0-6  Cross-sector rank (top sector = 6, bottom = 0)
      0-5  Rotation signal (into=5, accel=3, decel=1, out=0)
      0-4  Intra-sector position (sector_rank_pct / 25)
    """
    df = df.copy()
    df['sector_rank_score'] = 0.0

    for idx in df.index:
        score = 0.0

        # Cross-sector rank
        cross_rank = df.loc[idx, 'cross_sector_rank'] if 'cross_sector_rank' in df.columns else None
        if cross_rank is not None and not pd.isna(cross_rank):
            # Invert: rank 1 = best = 6 pts, rank 7 = worst = 0 pts
            score += max(0, 7 - float(cross_rank))

        # Rotation signal
        rotation = df.loc[idx, 'sector_rotation_signal'] if 'sector_rotation_signal' in df.columns else ''
        rot_scores = {'into': 5, 'accelerating': 3, 'decelerating': 1, 'out': 0, 'neutral': 2}
        score += rot_scores.get(str(rotation), 1)

        # Intra-sector percentile
        sector_pctile = df.loc[idx, 'sector_rank_pct'] if 'sector_rank_pct' in df.columns else None
        if sector_pctile is not None and not pd.isna(sector_pctile):
            score += min(float(sector_pctile) / 25, 4.0)

        df.loc[idx, 'sector_rank_score'] = round(min(score, 15.0), 1)

    return df


# ══════════════════════════════════════════════════════════════
# 4. THEME MOMENTUM SCORE (0-10)
# ══════════════════════════════════════════════════════════════

def compute_theme_momentum_score(df, sector_ts):
    """
    Map sector theme momentum into a per-row 0-10 score.

    Uses the latest theme_rank and rank_change_20d for each sector.
    """
    df = df.copy()
    df['theme_momentum_score'] = 5.0  # Default neutral

    if sector_ts is None or sector_ts.empty:
        return df

    # Get latest theme data per sector
    latest = sector_ts.sort_values('date').groupby('sector').tail(1)
    theme_map = {}
    for _, row in latest.iterrows():
        sector = row['sector']
        rank = row.get('theme_rank', 50)
        chg = row.get('rank_change_20d', 0)
        if pd.isna(chg):
            chg = 0

        # Score: 50% from current rank, 50% from momentum (change)
        rank_component = (rank / 100) * 5  # 0-5
        chg_component = max(0, min(5, (chg + 50) / 20))  # Normalize -50..+50 to 0..5
        theme_map[sector] = round(rank_component + chg_component, 1)

    # Map sectors to tickers
    if 'sector' in df.columns:
        for idx in df.index:
            sector = df.loc[idx, 'sector']
            if sector and not pd.isna(sector):
                df.loc[idx, 'theme_momentum_score'] = theme_map.get(str(sector), 5.0)
    else:
        # Try to assign from symbol
        for idx in df.index:
            sym = df.loc[idx, 'symbol'] if 'symbol' in df.columns else ''
            sector = _get_sector(sym)
            if sector != 'ETF':
                df.loc[idx, 'theme_momentum_score'] = theme_map.get(sector, 5.0)

    return df


# ══════════════════════════════════════════════════════════════
# 5. CONFLUENCE BONUS (0-10)
# ══════════════════════════════════════════════════════════════

def compute_confluence_bonus(df):
    """
    Bonus points when multiple signals agree.

    +3  Stage + strength aligned (Stage 2 + leader/outperformer, or Stage 4 + laggard/dog)
    +2  Sector rotation + event direction (sell in rotating-out sector, buy in rotating-in)
    +2  Event + IV regime aligned (bearish event + rich IV = put spread, bullish + cheap = calls)
    +2  Theme momentum + direction (rising theme + bullish, falling + bearish)
    +1  RS at 20d high + bullish event
    """
    df = df.copy()
    df['confluence_bonus'] = 0.0

    for idx in df.index:
        bonus = 0.0

        stage = df.loc[idx, 'w_stage'] if 'w_stage' in df.columns else None
        strength = df.loc[idx, 'abs_strength_label'] if 'abs_strength_label' in df.columns else ''
        rotation = str(df.loc[idx, 'sector_rotation_signal']) if 'sector_rotation_signal' in df.columns else ''
        iv_regime = df.loc[idx, 'iv_regime'] if 'iv_regime' in df.columns else ''
        direction = df.loc[idx, 'direction'] if 'direction' in df.columns else ''
        sell = df.loc[idx, 'sell_signal'] if 'sell_signal' in df.columns else False
        theme_score = df.loc[idx, 'theme_momentum_score'] if 'theme_momentum_score' in df.columns else 5
        event_code = str(df.loc[idx, 'event_code']) if 'event_code' in df.columns else 'none'

        is_bullish = direction == 'bullish' and not sell
        is_bearish = direction == 'bearish' or sell

        if pd.isna(stage):
            stage = None

        # Stage + strength aligned
        if stage == 2 and strength in ('leader', 'outperformer'):
            bonus += 3
        elif stage == 4 and strength in ('laggard', 'dog'):
            bonus += 3
        elif stage == 3 and strength in ('laggard', 'dog'):
            bonus += 2

        # Sector rotation + event direction
        if is_bearish and rotation == 'out':
            bonus += 2
        elif is_bullish and rotation in ('into', 'accelerating'):
            bonus += 2

        # Event + IV regime aligned
        if is_bearish and iv_regime == 'rich' and 'spread' in str(df.loc[idx, 'structure'] if 'structure' in df.columns else '').lower():
            bonus += 2
        elif is_bullish and iv_regime == 'cheap':
            bonus += 2

        # Theme momentum + direction
        if not pd.isna(theme_score):
            if is_bullish and float(theme_score) >= 7:
                bonus += 2
            elif is_bearish and float(theme_score) <= 3:
                bonus += 2

        # RS at 20d high + bullish
        rs_high = df.loc[idx, 'rs_at_20d_high'] if 'rs_at_20d_high' in df.columns else None
        if rs_high and is_bullish:
            bonus += 1

        df.loc[idx, 'confluence_bonus'] = round(min(bonus, 10.0), 1)

    return df


# ══════════════════════════════════════════════════════════════
# 6. MASTER SCORE (0-100)
# ══════════════════════════════════════════════════════════════

def compute_master_score(df):
    """
    Combine all components into master_score (0-100):

      30% — event_score (0-12 → 0-30)
      20% — tradeability_score (0-20 → 0-20)
      15% — abs_strength_score (0-10 → 0-15)
      15% — sector_rank_score (0-15 → 0-15)
      10% — theme_momentum_score (0-10 → 0-10)
      10% — confluence_bonus (0-10 → 0-10)
    """
    df = df.copy()

    event = df.get('event_score', pd.Series(0, index=df.index)).fillna(0)
    trade = df.get('tradeability_score', pd.Series(0, index=df.index)).fillna(0)
    strength = df.get('abs_strength_score', pd.Series(0, index=df.index)).fillna(0)
    sector = df.get('sector_rank_score', pd.Series(0, index=df.index)).fillna(0)
    theme = df.get('theme_momentum_score', pd.Series(5, index=df.index)).fillna(5)
    confluence = df.get('confluence_bonus', pd.Series(0, index=df.index)).fillna(0)

    df['master_score'] = (
        (event / 12 * 30).clip(0, 30) +
        (trade / 20 * 20).clip(0, 20) +
        (strength / 10 * 15).clip(0, 15) +
        (sector / 15 * 15).clip(0, 15) +
        (theme / 10 * 10).clip(0, 10) +
        (confluence / 10 * 10).clip(0, 10)
    ).round(1)

    # Component breakdown string
    df['master_breakdown'] = (
        'E=' + (event / 12 * 30).clip(0, 30).round(0).astype(int).astype(str) +
        ' T=' + (trade / 20 * 20).clip(0, 20).round(0).astype(int).astype(str) +
        ' S=' + (strength / 10 * 15).clip(0, 15).round(0).astype(int).astype(str) +
        ' R=' + (sector / 15 * 15).clip(0, 15).round(0).astype(int).astype(str) +
        ' M=' + (theme / 10 * 10).clip(0, 10).round(0).astype(int).astype(str) +
        ' C=' + (confluence / 10 * 10).clip(0, 10).round(0).astype(int).astype(str)
    )

    return df


# ══════════════════════════════════════════════════════════════
# 7. MASTER MODE — FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_master_mode(df, symbol_col='symbol', date_col='date',
                    open_col='open', high_col='high', low_col='low',
                    close_col='close', volume_col='volume'):
    """
    Full master pipeline:
      1. Stage-enriched scan (events + stages + strength + sector + IV + options)
      2. Theme momentum timeseries
      3. Tradeability scoring
      4. Sector rank scoring
      5. Theme momentum scoring + crossover detection
      6. Confluence bonus
      7. Master score computation
    """
    # ── Step 1: Stage-enriched scan ──
    result, regime, stages = stage_enriched_scan(
        df, symbol_col, date_col, open_col,
        high_col, low_col, close_col, volume_col
    )

    # ── Step 2: Theme momentum ──
    print("\nComputing theme momentum timeseries...")
    sector_ts, _, _ = theme_momentum_timeseries(
        df, symbol_col, date_col, close_col, outdir='.', stem='_master_theme'
    )

    # ── Step 3: Tradeability ──
    print("Computing tradeability scores...")
    result = compute_tradeability(result)

    # ── Step 4: Sector rank score ──
    print("Computing sector rank scores...")
    result = compute_sector_rank_score(result)

    # ── Step 5: Theme momentum score + crossovers ──
    print("Computing theme momentum scores...")
    result = compute_theme_momentum_score(result, sector_ts)

    crossovers = detect_theme_crossovers(sector_ts)
    if crossovers:
        print(f"\n  Theme crossovers detected:")
        for sector, info in crossovers.items():
            print(f"    {sector:22s} {info['direction']:12s} "
                  f"({info['rank_before']:.0f}% -> {info['rank_after']:.0f}%)")

    # ── Step 6: Confluence ──
    print("Computing confluence bonuses...")
    result = compute_confluence_bonus(result)

    # ── Step 7: Master score ──
    print("Computing master scores...")
    result = compute_master_score(result)

    # Sort by master score
    result = result.sort_values('master_score', ascending=False)

    # ── Print results (only tradeable) ──
    events = result[result.get('event_code', pd.Series('none')) != 'none']
    if 'tradeability_flag' in events.columns:
        tradeable = events['tradeability_flag'].fillna(False).sum()
        print(f"\n  TRADEABILITY GATE: {tradeable} tradeable / {len(events)} total events")
        events = events[events['tradeability_flag'].fillna(False)]

    print(f"\n{'='*100}")
    print(f"  MASTER SCORE LEADERBOARD — Top 30 Tradeable Signals")
    print(f"{'='*100}")
    print(f"  {'Sym':6s} {'Date':>11s} {'Master':>7s} {'Event':>22s} "
          f"{'Stage':>22s} {'Breakdown':>30s} {'Sell':>5s}")
    print(f"  {'-'*95}")

    for _, row in events.head(30).iterrows():
        sym = row.get('symbol', '?')
        date = str(row.get('date', ''))[:10]
        master = row.get('master_score', 0)
        event = str(row.get('event_label', ''))[:22]
        stage = str(row.get('w_stage_label', ''))[:22]
        breakdown = row.get('master_breakdown', '')
        sell = 'SELL' if row.get('sell_signal') else ''

        print(f"  {sym:6s} {date:>11s} {master:6.1f} {event:>22s} "
              f"{stage:>22s} {breakdown:>30s} {sell:>5s}")

    print(f"{'='*100}")

    # Score distribution
    if 'master_score' in events.columns:
        print(f"\n  Master score distribution (events only):")
        bins = [(80, 100, 'ELITE'), (60, 80, 'STRONG'), (40, 60, 'MODERATE'),
                (20, 40, 'WEAK'), (0, 20, 'NOISE')]
        for low, high, label in bins:
            count = ((events['master_score'] >= low) & (events['master_score'] < high)).sum()
            if count > 0:
                bar = '#' * min(count, 50)
                print(f"    {label:10s} ({low:>2d}-{high:>3d}): {count:4d}  {bar}")

    return result, regime, stages, sector_ts, crossovers


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 15: Running master score pipeline...")
    df = pd.read_csv('ohlcv.csv')
    result, regime, stages, sector_ts, crossovers = run_master_mode(df)

    result.to_csv('master_results.csv', index=False)
    print(f"\nSaved to master_results.csv")
