"""
Patch 12: Comparative Strength Analysis
Adds cross-sectional and intra-sector relative strength rankings
with rotation detection.

Metrics per ticker:
  sector                     — GICS sector assignment
  sector_rank                — rank within sector (1 = strongest)
  sector_rank_pct            — percentile within sector (0-99)
  sector_avg_perf_20d        — sector average 20d performance
  vs_sector_20d              — excess return vs own sector
  cross_sector_rank          — rank across all sectors (which sector is leading)
  sector_rotation_signal     — 'into' (money flowing in), 'out' (money leaving), 'neutral'
  pair_spread_vs_spy_20d     — alpha over SPY, adjusted for beta
  relative_strength_regime   — 'absolute_leader', 'sector_leader', 'sector_inline',
                                'sector_laggard', 'absolute_laggard'

Sector rotation detection:
  Compares sector average performance at 5d, 10d, 20d windows.
  'into' = sector outperforming universe avg AND accelerating (5d > 20d)
  'out'  = sector underperforming AND decelerating (5d < 20d)

Integration:
  Called from stage_enriched_scan after absolute strength,
  before options overlay. Adds sector context to every signal.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("comparative_strength")

# ── GICS Sector Mapping (expanded for common tickers) ──
_SECTOR_MAP = {
    # Tech / Semis
    'NVDA': 'Technology', 'AMD': 'Technology', 'AVGO': 'Technology',
    'INTC': 'Technology', 'ANET': 'Technology', 'ARM': 'Technology',
    'MSFT': 'Technology', 'AAPL': 'Technology', 'ORCL': 'Technology',
    # Software / Cloud
    'PLTR': 'Software', 'CRWD': 'Software', 'NET': 'Software',
    'DDOG': 'Software', 'SNOW': 'Software', 'PANW': 'Software',
    'ZS': 'Software', 'SHOP': 'Software', 'TTD': 'Software',
    # Internet / Media
    'META': 'Internet', 'GOOGL': 'Internet', 'AMZN': 'Internet',
    'NFLX': 'Internet', 'UBER': 'Internet', 'DASH': 'Internet',
    'RBLX': 'Internet',
    # EV / Mobility
    'TSLA': 'EV / Mobility', 'RIVN': 'EV / Mobility', 'LCID': 'EV / Mobility',
    'JOBY': 'EV / Mobility', 'ACHR': 'EV / Mobility',
    # Crypto / Fintech
    'COIN': 'Crypto / Fintech', 'MSTR': 'Crypto / Fintech',
    'SOFI': 'Crypto / Fintech', 'HOOD': 'Crypto / Fintech',
    # Nuclear / Energy
    'OKLO': 'Nuclear / Energy', 'VST': 'Nuclear / Energy',
    'CEG': 'Nuclear / Energy', 'GEV': 'Nuclear / Energy',
    'NNE': 'Nuclear / Energy', 'SMR': 'Nuclear / Energy',
    'WOLF': 'Nuclear / Energy',
    # Space / Quantum
    'LUNR': 'Space / Quantum', 'RKLB': 'Space / Quantum',
    'IONQ': 'Space / Quantum', 'RGTI': 'Space / Quantum',
    # Consumer
    'CELH': 'Consumer',
    # Hardware
    'SMCI': 'Hardware',
    # ETFs
    'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'TLT': 'ETF',
}


def _get_sector(symbol):
    return _SECTOR_MAP.get(symbol, 'Other')


def compute_comparative_strength(ohlcv_df, symbol_col='symbol', date_col='date',
                                  close_col='close'):
    """
    Compute cross-sectional and intra-sector relative strength.

    Returns:
      (ticker_results, sector_results)

      ticker_results: dict {symbol: {sector, sector_rank, sector_rank_pct,
                                      vs_sector_20d, relative_strength_regime, ...}}
      sector_results: dict {sector: {avg_perf_5d, avg_perf_10d, avg_perf_20d,
                                      cross_sector_rank, rotation_signal, members}}
    """
    df = ohlcv_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])

    # ── Per-ticker performance ──
    ticker_perf = {}
    close_arrays = {}

    for sym, g in df.groupby(symbol_col):
        close = g[close_col].values
        close_arrays[sym] = close
        n = len(close)

        if n < 30:
            continue

        last = float(close[-1])
        perf = {}
        for w, label in [(5, '5d'), (10, '10d'), (20, '20d'), (60, '60d')]:
            if n > w:
                perf[label] = round(((last / float(close[-(w+1)])) - 1) * 100, 2)
            else:
                perf[label] = None

        perf['sector'] = _get_sector(sym)
        ticker_perf[sym] = perf

    # ── Sector aggregation ──
    sector_tickers = {}
    for sym, perf in ticker_perf.items():
        sector = perf['sector']
        if sector == 'ETF':
            continue
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append((sym, perf))

    sector_results = {}
    for sector, members in sector_tickers.items():
        perfs_5d = [m[1].get('5d') for m in members if m[1].get('5d') is not None]
        perfs_10d = [m[1].get('10d') for m in members if m[1].get('10d') is not None]
        perfs_20d = [m[1].get('20d') for m in members if m[1].get('20d') is not None]

        avg_5d = round(np.mean(perfs_5d), 2) if perfs_5d else None
        avg_10d = round(np.mean(perfs_10d), 2) if perfs_10d else None
        avg_20d = round(np.mean(perfs_20d), 2) if perfs_20d else None

        sector_results[sector] = {
            'avg_perf_5d': avg_5d,
            'avg_perf_10d': avg_10d,
            'avg_perf_20d': avg_20d,
            'member_count': len(members),
            'members': [m[0] for m in members],
        }

    # ── Universe average (for rotation signal) ──
    all_perfs_5d = [v.get('5d') for v in ticker_perf.values()
                    if v.get('5d') is not None and v['sector'] != 'ETF']
    all_perfs_20d = [v.get('20d') for v in ticker_perf.values()
                     if v.get('20d') is not None and v['sector'] != 'ETF']
    universe_avg_5d = np.mean(all_perfs_5d) if all_perfs_5d else 0
    universe_avg_20d = np.mean(all_perfs_20d) if all_perfs_20d else 0

    # ── Cross-sector ranking (by 20d avg perf) ──
    ranked_sectors = sorted(
        [(s, d.get('avg_perf_20d', -999)) for s, d in sector_results.items()],
        key=lambda x: x[1], reverse=True
    )
    for rank_idx, (sector, _) in enumerate(ranked_sectors):
        sector_results[sector]['cross_sector_rank'] = rank_idx + 1

        avg_5d = sector_results[sector].get('avg_perf_5d') or 0
        avg_20d = sector_results[sector].get('avg_perf_20d') or 0

        # Rotation signal
        outperforming = avg_20d > universe_avg_20d
        accelerating = avg_5d > avg_20d

        if outperforming and accelerating:
            rotation = 'into'
        elif not outperforming and not accelerating:
            rotation = 'out'
        elif outperforming and not accelerating:
            rotation = 'decelerating'
        elif not outperforming and accelerating:
            rotation = 'accelerating'
        else:
            rotation = 'neutral'

        sector_results[sector]['rotation_signal'] = rotation

    # ── Intra-sector ranking ──
    ticker_results = {}

    for sector, members in sector_tickers.items():
        # Sort members by 20d perf within sector
        valid = [(sym, p) for sym, p in members if p.get('20d') is not None]
        valid.sort(key=lambda x: x[1]['20d'], reverse=True)

        n_members = len(valid)
        sector_avg_20d = sector_results[sector].get('avg_perf_20d', 0) or 0

        for rank_idx, (sym, perf) in enumerate(valid):
            sector_rank = rank_idx + 1
            sector_rank_pct = round((1 - rank_idx / max(n_members - 1, 1)) * 99, 0) if n_members > 1 else 50

            perf_20d = perf.get('20d', 0) or 0
            vs_sector = round(perf_20d - sector_avg_20d, 2)

            # RS vs SPY
            spy_close = close_arrays.get('SPY')
            pair_spread = None
            if spy_close is not None and sym in close_arrays:
                sc = close_arrays[sym]
                if len(sc) > 20 and len(spy_close) > 20:
                    stock_ret = (float(sc[-1]) / float(sc[-21])) - 1
                    spy_ret = (float(spy_close[-1]) / float(spy_close[-21])) - 1
                    pair_spread = round((stock_ret - spy_ret) * 100, 2)

            # Relative strength regime
            abs_score = None
            for s2, p2 in members:
                if s2 == sym:
                    break

            # Determine regime based on universe rank + sector rank
            regime = _classify_rs_regime(
                sector_rank_pct=sector_rank_pct,
                vs_sector=vs_sector,
                pair_spread=pair_spread,
                sector_rotation=sector_results[sector].get('rotation_signal', 'neutral'),
            )

            ticker_results[sym] = {
                'sector': sector,
                'sector_rank': sector_rank,
                'sector_rank_pct': sector_rank_pct,
                'sector_size': n_members,
                'sector_avg_perf_20d': sector_avg_20d,
                'vs_sector_20d': vs_sector,
                'cross_sector_rank': sector_results[sector].get('cross_sector_rank'),
                'sector_rotation_signal': sector_results[sector].get('rotation_signal'),
                'pair_spread_vs_spy_20d': pair_spread,
                'relative_strength_regime': regime,
            }

    return ticker_results, sector_results


def _classify_rs_regime(sector_rank_pct, vs_sector, pair_spread, sector_rotation):
    """
    Classify into:
      absolute_leader   — top of strong sector, outperforming SPY
      sector_leader     — top of sector, but sector may be weak
      sector_inline     — middle of sector
      sector_laggard    — bottom of sector
      absolute_laggard  — bottom of weak sector, underperforming SPY
    """
    is_sector_top = sector_rank_pct >= 70
    is_sector_bottom = sector_rank_pct <= 30
    beats_spy = (pair_spread or 0) > 2
    trails_spy = (pair_spread or 0) < -2
    sector_strong = sector_rotation in ('into', 'decelerating')
    sector_weak = sector_rotation in ('out',)

    if is_sector_top and beats_spy and sector_strong:
        return 'absolute_leader'
    elif is_sector_top:
        return 'sector_leader'
    elif is_sector_bottom and trails_spy and sector_weak:
        return 'absolute_laggard'
    elif is_sector_bottom:
        return 'sector_laggard'
    else:
        return 'sector_inline'


def apply_comparative_strength(df, ohlcv_df, symbol_col='symbol', date_col='date',
                                close_col='close'):
    """
    Apply comparative strength overlay to an enriched scan DataFrame.
    Merges sector rankings, rotation signals, and RS regime into every row.
    """
    ticker_results, sector_results = compute_comparative_strength(
        ohlcv_df, symbol_col, date_col, close_col
    )

    # Build merge DataFrame
    rows = []
    for sym, data in ticker_results.items():
        row = {symbol_col: sym}
        row.update(data)
        rows.append(row)

    if not rows:
        return df

    comp_df = pd.DataFrame(rows)
    merge_cols = [c for c in comp_df.columns if c != symbol_col]

    # Drop existing columns to avoid conflicts
    for col in merge_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(comp_df, on=symbol_col, how='left')

    # ── Print sector rotation summary ──
    print(f"\n  Sector Rotation ({len(sector_results)} sectors):")
    print(f"  {'Sector':22s} {'Rank':>5s} {'5d':>7s} {'10d':>7s} {'20d':>7s} {'#':>3s} {'Rotation':>14s}")
    print(f"  {'-'*72}")

    sorted_sectors = sorted(sector_results.items(),
                            key=lambda x: x[1].get('cross_sector_rank', 99))
    for sector, data in sorted_sectors:
        rotation = data.get('rotation_signal', '')
        rot_marker = {
            'into': '>>> INTO',
            'out': '<<< OUT',
            'accelerating': '^ ACCEL',
            'decelerating': 'v DECEL',
            'neutral': '= NEUTRAL',
        }.get(rotation, rotation)

        print(f"  {sector:22s} "
              f"#{data.get('cross_sector_rank', '?'):>3} "
              f"{data.get('avg_perf_5d', 0):>+6.1f}% "
              f"{data.get('avg_perf_10d', 0):>+6.1f}% "
              f"{data.get('avg_perf_20d', 0):>+6.1f}% "
              f"{data.get('member_count', 0):>3d} "
              f"{rot_marker:>14s}")

    # ── Print RS regime distribution ──
    if 'relative_strength_regime' in df.columns:
        regime_dist = df.drop_duplicates(subset=[symbol_col])['relative_strength_regime'].value_counts()
        print(f"\n  RS Regime distribution:")
        for regime, count in regime_dist.items():
            bar = '#' * (count * 2)
            print(f"    {regime:22s}: {count:3d}  {bar}")

    # ── Sector leaders ──
    if 'sector_rank' in df.columns:
        leaders = df.drop_duplicates(subset=[symbol_col])
        leaders = leaders[leaders['sector_rank'] == 1]
        if not leaders.empty:
            print(f"\n  Sector Leaders (rank #1 in each sector):")
            for _, row in leaders.iterrows():
                print(f"    {row[symbol_col]:6s}  {row.get('sector', ''):22s}  "
                      f"vs_sector: {row.get('vs_sector_20d', 0):+.1f}%  "
                      f"RS regime: {row.get('relative_strength_regime', '')}")

    logger.info(f"Comparative strength applied: {len(ticker_results)} tickers, {len(sector_results)} sectors")
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 12: Testing comparative strength...")
    df = pd.read_csv('ohlcv.csv')
    df['date'] = pd.to_datetime(df['date'])

    ticker_results, sector_results = compute_comparative_strength(df)

    print(f"\n{'='*80}")
    print(f"  SECTOR ROTATION TABLE")
    print(f"{'='*80}")
    print(f"  {'Sector':22s} {'Rank':>5s} {'5d':>7s} {'10d':>7s} {'20d':>7s} {'#':>3s} {'Rotation':>14s}")
    print(f"  {'-'*72}")

    sorted_sectors = sorted(sector_results.items(),
                            key=lambda x: x[1].get('cross_sector_rank', 99))
    for sector, data in sorted_sectors:
        rotation = data.get('rotation_signal', '')
        print(f"  {sector:22s} "
              f"#{data.get('cross_sector_rank', '?'):>3} "
              f"{data.get('avg_perf_5d', 0):>+6.1f}% "
              f"{data.get('avg_perf_10d', 0):>+6.1f}% "
              f"{data.get('avg_perf_20d', 0):>+6.1f}% "
              f"{data.get('member_count', 0):>3d} "
              f"{rotation:>14s}")

    print(f"\n{'='*80}")
    print(f"  INTRA-SECTOR RANKINGS")
    print(f"{'='*80}")
    for sector in sorted(sector_results.keys()):
        members = [(sym, d) for sym, d in ticker_results.items() if d['sector'] == sector]
        members.sort(key=lambda x: x[1]['sector_rank'])
        print(f"\n  {sector} ({len(members)} members):")
        for sym, d in members:
            print(f"    #{d['sector_rank']:2d}  {sym:6s}  "
                  f"vs_sector: {d.get('vs_sector_20d', 0):>+6.1f}%  "
                  f"vs_SPY: {d.get('pair_spread_vs_spy_20d', 0):>+6.1f}%  "
                  f"regime: {d['relative_strength_regime']}")
