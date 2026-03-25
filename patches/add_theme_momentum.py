"""
Patch 14: Theme Momentum Timeseries
Adds theme_momentum_timeseries() and --mode theme_momentum to CLI.

theme_momentum_timeseries():
  - Computes daily theme_score per sector (median of member multi-horizon momentum)
  - Ranks sectors cross-sectionally each day
  - 5-day rolling smoothed theme score
  - Plotly line chart: theme rank over time for top 8 sectors
  - CSV timeseries export

Key insight:
  theme_tracker (patch 12) shows the current snapshot — which sectors lead NOW.
  theme_momentum shows the trajectory — which sectors are GAINING vs LOSING.
  A sector rising from 40th to 80th percentile over 3 weeks = rotation target.
  A sector falling from 90th to 60th = rotation away, tighten stops.

CLI: --mode theme_momentum
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("theme_momentum")

# Reuse sector map from comparative strength
from patches.add_comparative_strength import _SECTOR_MAP, _get_sector


def theme_momentum_timeseries(df, symbol_col='symbol', date_col='date',
                               close_col='close', outdir='output',
                               stem='theme_momentum'):
    """
    Compute daily theme momentum timeseries per sector.

    Theme score = 0.5 * 21d return + 0.3 * 63d return + 0.2 * 126d return
    (weighted multi-horizon momentum, heavier on recent)

    Returns (sector_ts DataFrame, csv_path, html_path)
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([symbol_col, date_col])

    # Assign sectors
    d['sector'] = d[symbol_col].map(_get_sector).fillna('Other')

    # Skip ETFs
    d = d[d['sector'] != 'ETF'].copy()

    # Per-ticker multi-horizon returns
    d['ret_21d'] = d.groupby(symbol_col)[close_col].pct_change(21)
    d['ret_63d'] = d.groupby(symbol_col)[close_col].pct_change(63)
    d['ret_126d'] = d.groupby(symbol_col)[close_col].pct_change(126)

    # Weighted theme score
    d['theme_score'] = (
        0.5 * d['ret_21d'].fillna(0) +
        0.3 * d['ret_63d'].fillna(0) +
        0.2 * d['ret_126d'].fillna(0)
    )

    # Sector-level daily aggregation (median of member scores)
    sector_ts = d.groupby([date_col, 'sector'], as_index=False).agg(
        theme_score=('theme_score', 'median'),
        member_count=('theme_score', 'count'),
    )

    # Cross-sectional rank each day (percentile)
    sector_ts['theme_rank'] = sector_ts.groupby(date_col)['theme_score'].rank(
        pct=True, method='average'
    ) * 100

    # 5-day rolling smoothed score
    sector_ts = sector_ts.sort_values([date_col, 'sector'])
    sector_ts['theme_roll_5'] = sector_ts.groupby('sector')['theme_score'].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # Rank change (vs 20 trading days ago)
    sector_ts['theme_rank_20d_ago'] = sector_ts.groupby('sector')['theme_rank'].shift(20)
    sector_ts['rank_change_20d'] = sector_ts['theme_rank'] - sector_ts['theme_rank_20d_ago']

    # ── Output directory ──
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # ── CSV export ──
    csv_path = outdir / f'{stem}_timeseries.csv'
    sector_ts.to_csv(csv_path, index=False)
    logger.info(f"Timeseries CSV saved: {csv_path}")

    # ── Plotly chart ──
    html_path = None
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Get top 8 sectors by latest rank
        latest = sector_ts.sort_values(date_col).groupby('sector').tail(1)
        top_sectors = latest.nlargest(8, 'theme_rank')['sector'].tolist()

        plot_data = sector_ts[sector_ts['sector'].isin(top_sectors)].copy()

        fig = px.line(
            plot_data,
            x=date_col,
            y='theme_rank',
            color='sector',
            title='Theme Rank Over Time (Top Sectors)',
            template='plotly_dark',
            labels={'theme_rank': 'Theme Rank (percentile)', date_col: 'Date'},
        )

        # Add 50th percentile reference line
        fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.4,
                       annotation_text='50th pctile')

        fig.update_layout(
            font=dict(family='Consolas', size=12),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            yaxis=dict(range=[0, 105]),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.25,
                xanchor='center',
                x=0.5,
            ),
        )

        html_path = outdir / f'{stem}_chart.html'
        fig.write_html(str(html_path))
        logger.info(f"Chart saved: {html_path}")

    except ImportError:
        logger.warning("Plotly not installed — skipping chart")

    return sector_ts, csv_path, html_path


def print_theme_momentum_summary(sector_ts, date_col='date'):
    """Print the latest theme momentum snapshot + biggest movers."""
    latest = sector_ts.sort_values(date_col).groupby('sector').tail(1).copy()
    latest = latest.sort_values('theme_rank', ascending=False)

    print(f"\n{'='*80}")
    print(f"  THEME MOMENTUM — Latest Snapshot")
    print(f"{'='*80}")
    print(f"  {'Sector':22s} {'Rank':>6s} {'Score':>8s} {'Roll5':>8s} {'Chg20d':>8s} {'#':>3s} {'Trend':>10s}")
    print(f"  {'-'*72}")

    for _, row in latest.iterrows():
        rank = row.get('theme_rank', 0)
        score = row.get('theme_score', 0)
        roll5 = row.get('theme_roll_5', 0)
        chg = row.get('rank_change_20d')
        n = row.get('member_count', 0)

        # Trend arrow
        if chg is not None and not pd.isna(chg):
            if chg > 10:
                trend = '>>> RISING'
            elif chg > 3:
                trend = '^ up'
            elif chg < -10:
                trend = '<<< FALLING'
            elif chg < -3:
                trend = 'v down'
            else:
                trend = '= flat'
            chg_str = f'{chg:+.0f}'
        else:
            trend = '? new'
            chg_str = 'n/a'

        print(f"  {row['sector']:22s} "
              f"{rank:5.0f}% "
              f"{score:>+7.3f} "
              f"{roll5:>+7.3f} "
              f"{chg_str:>8s} "
              f"{int(n):>3d} "
              f"{trend:>10s}")

    print(f"{'='*80}")

    # Biggest movers (by 20d rank change)
    movers = latest.dropna(subset=['rank_change_20d'])
    if not movers.empty:
        risers = movers.nlargest(3, 'rank_change_20d')
        fallers = movers.nsmallest(3, 'rank_change_20d')

        print(f"\n  Biggest 20d rank gainers:")
        for _, r in risers.iterrows():
            if r['rank_change_20d'] > 0:
                print(f"    {r['sector']:22s}  +{r['rank_change_20d']:.0f} pts")

        print(f"\n  Biggest 20d rank losers:")
        for _, r in fallers.iterrows():
            if r['rank_change_20d'] < 0:
                print(f"    {r['sector']:22s}  {r['rank_change_20d']:.0f} pts")


def run_theme_momentum_mode(ohlcv_df, outdir='output', stamp=None):
    """
    Full theme momentum pipeline for CLI mode.
    Returns (sector_ts, csv_path, html_path)
    """
    if stamp is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "=" * 60)
    print("  THEME MOMENTUM ANALYSIS")
    print("=" * 60)

    sector_ts, csv_path, html_path = theme_momentum_timeseries(
        ohlcv_df, outdir=outdir, stem=f'theme_momentum_{stamp}'
    )

    print_theme_momentum_summary(sector_ts)

    print(f"\n  Outputs:")
    print(f"    CSV: {csv_path}")
    if html_path:
        print(f"    Chart: {html_path}")

    return sector_ts, csv_path, html_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 14: Testing theme momentum...")
    df = pd.read_csv('ohlcv.csv')

    sector_ts, csv_path, html_path = run_theme_momentum_mode(df)
    print(f"\nTimeseries rows: {len(sector_ts)}")
    print(f"Sectors tracked: {sector_ts['sector'].nunique()}")
