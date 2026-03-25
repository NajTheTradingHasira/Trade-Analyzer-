"""
Patch 16: Master Matrix — Full Cross-Dimensional Signal Ranking
================================================================
Runs the complete pipeline and computes a master_matrix_score (0-100 percentile)
combining every analytical layer.

Pipeline:
  1. unified_event_scan (gaps, climax, trendline breaks, composite events)
  2. tradeability_composite_score
  3. sector_relative_rank_table (intra-sector + cross-sector)
  4. theme_crossover_alerts
  5. master_matrix_score computation
  6. Percentile ranking + tier bucketing

Score formula:
  Base (70 pts max):
    +25% event_score (normalized 0-12)
    +15% tradeability_score (normalized 0-20)
    +15% absolute_strength_rank (0-100 percentile)
    +15% comparative_strength_rank (sector_rank_pct 0-99)

  Penalties (up to -25 pts):
    -15% climax_score (climax top or trendline break severity)
    -10% gap_fill_risk (high fill risk = signal less reliable)

  Bonuses (up to +30 pts):
    +6  theme_crossover_into (sector crossing UP above 50th pctile)
    +5  sector_leader (rank #1 in sector)
    +5  absolute_strength_alert (AS tier 1/2/3)
    +5  post_earnings_flag (earnings gap = higher continuation)
    +4  tradeability_flag (tradeability >= 15)
    +3  confluence (stage + direction + sector aligned)
    +2  rs_at_20d_high (relative strength confirming)

  Final: percentile rank 0-100 across all signals in universe.

Tiers:
  Elite       (>=95th pctile)
  Very Strong (>=85th)
  Strong      (>=70th)
  Average     (>=40th)
  Below Avg   (>=20th)
  Weak        (>=5th)
  Very Weak   (<5th)

CLI: --mode master_matrix
Output: full CSV + top 50 CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from datetime import datetime

from patches.add_unified_scanner import unified_event_scan
from patches.add_stage_enriched import (
    classify_universe, get_market_regime, stage_enriched_scan
)
from patches.add_absolute_strength import compute_absolute_strength_universe
from patches.add_comparative_strength import compute_comparative_strength, _get_sector
from patches.add_theme_momentum import theme_momentum_timeseries
from patches.add_master_score import detect_theme_crossovers
from patches.options_overlay import apply_options_overlay
from patches.iv_provider import fetch_iv_universe
from patches.add_as_monitor import absolute_strength_monitor
from patches.add_master_score import compute_tradeability

logger = logging.getLogger("master_matrix")

TIER_LABELS = [
    (95, 'Elite'),
    (85, 'Very Strong'),
    (70, 'Strong'),
    (40, 'Average'),
    (20, 'Below Avg'),
    (5,  'Weak'),
    (0,  'Very Weak'),
]


def _assign_tier(pctile):
    for threshold, label in TIER_LABELS:
        if pctile >= threshold:
            return label
    return 'Very Weak'


def compute_climax_penalty(df):
    """
    Compute climax penalty (0-15) based on climax top / trendline break severity.
    """
    df = df.copy()
    df['climax_penalty'] = 0.0

    for idx in df.index:
        penalty = 0.0
        event_code = str(df.loc[idx, 'event_code']) if 'event_code' in df.columns else 'none'
        sell = df.loc[idx, 'sell_signal'] if 'sell_signal' in df.columns else False
        stage = df.loc[idx, 'w_stage'] if 'w_stage' in df.columns else None

        if event_code == 'climax_top':
            penalty = 12
            if stage == 4:
                penalty = 15  # Climax in Stage 4 = worst case
            elif stage == 3:
                penalty = 13
        elif event_code == 'trendline_break':
            penalty = 10
            if stage == 4:
                penalty = 12
        elif sell and event_code not in ('none', 'common_gap'):
            penalty = 5

        df.loc[idx, 'climax_penalty'] = penalty

    return df


def compute_fill_risk_penalty(df):
    """
    Compute gap fill risk penalty (0-10).
    High fill risk means the gap is likely to fill = signal less reliable.
    """
    df = df.copy()
    df['fill_risk_penalty'] = 0.0

    for idx in df.index:
        fill_risk = df.loc[idx, 'gap_fill_risk'] if 'gap_fill_risk' in df.columns else 'n/a'
        fill_dist = df.loc[idx, 'fill_distance_pct'] if 'fill_distance_pct' in df.columns else None

        if fill_risk == 'high':
            penalty = 8
            if fill_dist is not None and not pd.isna(fill_dist) and float(fill_dist) > 0.8:
                penalty = 10  # Nearly filled = very unreliable
        elif fill_risk == 'medium':
            penalty = 4
        else:
            penalty = 0

        df.loc[idx, 'fill_risk_penalty'] = penalty

    return df


def compute_bonuses(df, crossovers=None, as_monitor_results=None):
    """
    Compute bonus points (0-30 max) from multiple confirming signals.
    """
    df = df.copy()
    df['mm_bonus'] = 0.0
    df['mm_bonus_detail'] = ''

    # Build lookup maps
    cross_up_sectors = set()
    if crossovers:
        for sector, info in crossovers.items():
            if info['direction'] == 'CROSS_UP':
                cross_up_sectors.add(sector)

    as_tiers = {}
    if as_monitor_results:
        for sym, data in as_monitor_results.items():
            pctile = data.get('as_percentile_rank', 0)
            if pctile and pctile >= 80:
                as_tiers[sym] = pctile

    for idx in df.index:
        bonus = 0.0
        details = []
        sym = df.loc[idx, 'symbol'] if 'symbol' in df.columns else ''
        sector = df.loc[idx, 'sector'] if 'sector' in df.columns else _get_sector(sym)
        stage = df.loc[idx, 'w_stage'] if 'w_stage' in df.columns else None
        direction = df.loc[idx, 'direction'] if 'direction' in df.columns else ''
        sell = df.loc[idx, 'sell_signal'] if 'sell_signal' in df.columns else False
        is_bullish = direction == 'bullish' and not sell

        # +6 theme crossover into
        if sector in cross_up_sectors and is_bullish:
            bonus += 6
            details.append(f'theme_cross_up({sector})+6')

        # +5 sector leader
        sector_rank = df.loc[idx, 'sector_rank'] if 'sector_rank' in df.columns else None
        if sector_rank is not None and not pd.isna(sector_rank) and int(sector_rank) == 1:
            bonus += 5
            details.append('sector_leader+5')

        # +5 AS alert tier
        if sym in as_tiers:
            pctile = as_tiers[sym]
            if pctile >= 95:
                bonus += 5
                details.append('AS_elite+5')
            elif pctile >= 90:
                bonus += 4
                details.append('AS_very_strong+4')
            elif pctile >= 80:
                bonus += 3
                details.append('AS_strong+3')

        # +5 post earnings flag
        is_earnings = df.loc[idx, 'is_earnings_gap'] if 'is_earnings_gap' in df.columns else False
        if is_earnings and not pd.isna(is_earnings) and is_earnings:
            bonus += 5
            details.append('earnings_gap+5')

        # +4 tradeability flag
        trade_score = df.loc[idx, 'tradeability_score'] if 'tradeability_score' in df.columns else 0
        if trade_score and not pd.isna(trade_score) and float(trade_score) >= 15:
            bonus += 4
            details.append('high_tradeability+4')

        # +3 confluence (stage + direction + sector rotation aligned)
        rotation = str(df.loc[idx, 'sector_rotation_signal']) if 'sector_rotation_signal' in df.columns else ''
        if pd.isna(stage):
            stage = None
        if stage == 2 and is_bullish and rotation in ('into', 'accelerating'):
            bonus += 3
            details.append('confluence_bull+3')
        elif stage == 4 and sell and rotation == 'out':
            bonus += 3
            details.append('confluence_bear+3')

        # +2 RS at 20d high
        rs_high = df.loc[idx, 'rs_at_20d_high'] if 'rs_at_20d_high' in df.columns else None
        if rs_high and is_bullish:
            bonus += 2
            details.append('rs_20d_high+2')

        df.loc[idx, 'mm_bonus'] = round(min(bonus, 30.0), 1)
        df.loc[idx, 'mm_bonus_detail'] = '; '.join(details) if details else ''

    return df


def compute_master_matrix_score(df):
    """
    Compute the master_matrix_raw_score and percentile rank.

    Raw score = base + bonuses - penalties
    Then ranked 0-100 percentile within the universe.
    """
    df = df.copy()

    # ── Base components (normalized) ──
    event = df.get('event_score', pd.Series(0, index=df.index)).fillna(0).astype(float)
    trade = df.get('tradeability_score', pd.Series(0, index=df.index)).fillna(0).astype(float)
    abs_rank = df.get('abs_rank_20d', pd.Series(50, index=df.index)).fillna(50).astype(float)
    sector_pct = df.get('sector_rank_pct', pd.Series(50, index=df.index)).fillna(50).astype(float)

    # Normalize to contribution weights
    event_contrib = (event / 12 * 25).clip(0, 25)       # 25% max
    trade_contrib = (trade / 20 * 15).clip(0, 15)       # 15% max
    abs_contrib = (abs_rank / 100 * 15).clip(0, 15)     # 15% max
    sector_contrib = (sector_pct / 100 * 15).clip(0, 15) # 15% max

    base = event_contrib + trade_contrib + abs_contrib + sector_contrib  # 0-70

    # ── Penalties ──
    climax_pen = df.get('climax_penalty', pd.Series(0, index=df.index)).fillna(0).astype(float)
    fill_pen = df.get('fill_risk_penalty', pd.Series(0, index=df.index)).fillna(0).astype(float)

    penalties = (climax_pen / 15 * 15).clip(0, 15) + (fill_pen / 10 * 10).clip(0, 10)  # 0-25

    # ── Bonuses ──
    bonuses = df.get('mm_bonus', pd.Series(0, index=df.index)).fillna(0).astype(float)  # 0-30

    # ── Raw score ──
    raw = (base + bonuses - penalties).clip(0, 100)
    df['mm_raw_score'] = raw.round(2)

    # ── Percentile rank ──
    df['master_matrix_score'] = df['mm_raw_score'].rank(pct=True).mul(100).round(1)

    # ── Tier ──
    df['mm_tier'] = df['master_matrix_score'].apply(_assign_tier)

    # ── Breakdown string ──
    df['mm_breakdown'] = (
        'E=' + event_contrib.round(0).astype(int).astype(str) +
        ' T=' + trade_contrib.round(0).astype(int).astype(str) +
        ' A=' + abs_contrib.round(0).astype(int).astype(str) +
        ' S=' + sector_contrib.round(0).astype(int).astype(str) +
        ' B=' + bonuses.round(0).astype(int).astype(str) +
        ' P=-' + penalties.round(0).astype(int).astype(str)
    )

    return df


def run_master_matrix(df, symbol_col='symbol', date_col='date',
                      open_col='open', high_col='high', low_col='low',
                      close_col='close', volume_col='volume'):
    """
    Full master matrix pipeline.
    """
    print("\n" + "=" * 80)
    print("  MASTER MATRIX — Full Cross-Dimensional Signal Ranking")
    print("=" * 80)

    # ── Step 1: Stage-enriched scan (events + stages + strength + sector + IV) ──
    result, regime, stages = stage_enriched_scan(
        df, symbol_col, date_col, open_col,
        high_col, low_col, close_col, volume_col
    )

    # ── Step 2: Theme momentum + crossovers ──
    print("\nComputing theme momentum...")
    sector_ts, _, _ = theme_momentum_timeseries(
        df, symbol_col, date_col, close_col, outdir='.', stem='_mm_theme'
    )
    crossovers = detect_theme_crossovers(sector_ts)

    if crossovers:
        print(f"\n  Theme crossovers:")
        for sector, info in crossovers.items():
            print(f"    {sector:22s} {info['direction']:12s} "
                  f"({info['rank_before']:.0f}% -> {info['rank_after']:.0f}%)")

    # ── Step 3: AS monitor (for bonus tiers) ──
    print("\nRunning AS monitor for bonus tiers...")
    as_results = absolute_strength_monitor(df, symbol_col, date_col, close_col, volume_col)

    # ── Step 4: Tradeability ──
    print("Computing tradeability scores...")
    result = compute_tradeability(result)

    # ── Step 5: Penalties ──
    print("Computing climax + fill risk penalties...")
    result = compute_climax_penalty(result)
    result = compute_fill_risk_penalty(result)

    # ── Step 6: Bonuses ──
    print("Computing bonuses (theme crossover, sector leader, AS alert, earnings, etc.)...")
    result = compute_bonuses(result, crossovers, as_results)

    # ── Step 7: Master matrix score ──
    print("Computing master matrix scores + percentile ranking...")
    result = compute_master_matrix_score(result)

    # Sort
    result = result.sort_values('master_matrix_score', ascending=False)

    # ── HARD FILTER: only trade tradeability_flag = True ──
    tradeable_mask = result.get('tradeability_flag', pd.Series(True, index=result.index)).fillna(False)
    n_before = len(result)
    n_filtered = (~tradeable_mask).sum()
    result['tradeable'] = tradeable_mask

    print(f"\n  TRADEABILITY GATE: {tradeable_mask.sum()} tradeable / {n_before} total ({n_filtered} filtered out)")

    # ── Print results ──
    events = result[
        (result.get('event_code', pd.Series('none')) != 'none') &
        (result.get('tradeable', pd.Series(True)))
    ].copy()

    print(f"\n{'='*110}")
    print(f"  MASTER MATRIX LEADERBOARD — Top 50")
    print(f"{'='*110}")
    print(f"  {'Sym':6s} {'Date':>11s} {'Pctile':>7s} {'Tier':>12s} {'Raw':>6s} "
          f"{'Event':>22s} {'Stage':>22s} "
          f"{'Breakdown':>28s} {'Bonuses':>30s}")
    print(f"  {'-'*108}")

    for _, row in events.head(50).iterrows():
        sym = row.get(symbol_col, '?')
        date = str(row.get(date_col, ''))[:10]
        pctile = row.get('master_matrix_score', 0)
        tier = row.get('mm_tier', '')
        raw = row.get('mm_raw_score', 0)
        event = str(row.get('event_label', ''))[:22]
        stage = str(row.get('w_stage_label', ''))[:22]
        breakdown = row.get('mm_breakdown', '')
        bonus_detail = str(row.get('mm_bonus_detail', ''))[:30]

        print(f"  {sym:6s} {date:>11s} {pctile:5.1f}% {tier:>12s} {raw:5.1f} "
              f"{event:>22s} {stage:>22s} "
              f"{breakdown:>28s} {bonus_detail:>30s}")

    print(f"{'='*110}")

    # ── Tier distribution ──
    if 'mm_tier' in events.columns:
        tier_dist = events['mm_tier'].value_counts()
        print(f"\n  Tier distribution ({len(events)} signals):")
        for tier_label in ['Elite', 'Very Strong', 'Strong', 'Average', 'Below Avg', 'Weak', 'Very Weak']:
            count = tier_dist.get(tier_label, 0)
            if count > 0:
                bar = '#' * min(count // 2 + 1, 50)
                print(f"    {tier_label:12s}: {count:5d}  {bar}")

    # ── Top signals per tier ──
    for tier_label in ['Elite', 'Very Strong']:
        tier_sigs = events[events['mm_tier'] == tier_label]
        if not tier_sigs.empty:
            print(f"\n  {tier_label.upper()} signals ({len(tier_sigs)}):")
            for _, row in tier_sigs.head(10).iterrows():
                sell_marker = ' [SELL]' if row.get('sell_signal') else ''
                print(f"    {row.get(symbol_col, '?'):6s} {str(row.get(date_col, ''))[:10]:>11s} "
                      f"pctile={row.get('master_matrix_score', 0):5.1f}% "
                      f"raw={row.get('mm_raw_score', 0):5.1f} "
                      f"{str(row.get('event_label', ''))[:30]}{sell_marker}")

    return result, regime, stages, crossovers


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 16: Running master matrix...")
    df = pd.read_csv('ohlcv.csv')
    result, regime, stages, crossovers = run_master_matrix(df)

    result.to_csv('master_matrix_full.csv', index=False)

    events = result[result.get('event_code', pd.Series('none')) != 'none']
    events.head(50).to_csv('master_matrix_top50.csv', index=False)

    print(f"\nSaved: master_matrix_full.csv ({len(result)} rows)")
    print(f"Saved: master_matrix_top50.csv ({min(50, len(events))} rows)")
