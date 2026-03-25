"""
Patch 18: Linear Trend Tracker
Fits log-linear regression over rolling lookback window per ticker.

Metrics:
  lin_slope          — annualized trend slope (log-space, *252)
  lin_r2             — goodness of fit (how "clean" the trend is)
  lin_pos            — total return over the lookback window
  linear_strength    — composite: slope*1000 + r2*10 + position
  linear_rank        — percentile rank within universe (0-100)
  linear_flag        — True if slope>0, R2>0.7, position>15%
  linear_label       — 'linear_leader' or 'neutral'

Key insight:
  R2 measures trend QUALITY, not direction.
  High R2 + positive slope = clean uptrend (Stage 2 institutional accumulation)
  High R2 + negative slope = clean downtrend (Stage 4)
  Low R2 = choppy (Stage 1 or 3 - no clean trend)

  Plotly scatter: R2 vs linear_rank (flagged leaders highlighted)

CLI: --mode linear_tracker
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("linear_tracker")


def linear_tracker(df, symbol_col='symbol', date_col='date', close_col='close',
                   lookback=126, outdir='.', stem='linear_tracker'):
    """
    Fit log-linear regression over rolling lookback window for each ticker.
    Returns (full_df, latest_snapshot, csv_path, html_path).
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([symbol_col, date_col])

    out = []

    for sym, g in d.groupby(symbol_col):
        g = g.copy().reset_index(drop=True)
        n = len(g)
        close = g[close_col].values.astype(float)

        # Replace zeros/negatives for log
        close = np.where(close <= 0, np.nan, close)
        log_close = np.log(close)

        lin_slope = np.full(n, np.nan)
        lin_r2 = np.full(n, np.nan)
        lin_pos = np.full(n, np.nan)

        if n >= lookback:
            for i in range(lookback - 1, n):
                yy = log_close[i - lookback + 1:i + 1]
                valid_mask = np.isfinite(yy)

                if valid_mask.sum() < lookback * 0.8:
                    continue

                yy_clean = yy[valid_mask]
                xx_clean = np.arange(valid_mask.sum())

                # Linear regression
                coef = np.polyfit(xx_clean, yy_clean, 1)
                pred = np.polyval(coef, xx_clean)

                ss_res = np.sum((yy_clean - pred) ** 2)
                ss_tot = np.sum((yy_clean - yy_clean.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                # Annualized slope (daily slope * 252)
                lin_slope[i] = coef[0] * 252
                lin_r2[i] = r2

                # Position: total return over lookback
                start_price = close[i - lookback + 1]
                end_price = close[i]
                if start_price and not np.isnan(start_price) and start_price > 0:
                    lin_pos[i] = (end_price / start_price) - 1

        g['lin_slope'] = lin_slope
        g['lin_r2'] = lin_r2
        g['lin_pos'] = lin_pos

        # Composite strength
        g['linear_strength'] = (
            np.nan_to_num(g['lin_slope'].values.copy(), 0) * 1000 +
            np.nan_to_num(g['lin_r2'].values.copy(), 0) * 10 +
            np.nan_to_num(g['lin_pos'].values.copy(), 0)
        )

        # Flag: strong clean uptrend
        g['linear_flag'] = (
            (g['lin_slope'] > 0) &
            (g['lin_r2'] > 0.7) &
            (g['lin_pos'] > 0.15)
        ).fillna(False)

        g['linear_label'] = np.where(g['linear_flag'], 'linear_leader', 'neutral')
        g[symbol_col] = sym
        out.append(g)

    if not out:
        return pd.DataFrame(), pd.DataFrame(), None, None

    full = pd.concat(out, ignore_index=True)

    # Latest snapshot per ticker
    latest = full.sort_values(date_col).groupby(symbol_col, as_index=False).tail(1)

    # Universe rank on latest snapshot
    valid_strength = latest['linear_strength'].dropna()
    if len(valid_strength) > 1:
        latest['linear_rank'] = latest['linear_strength'].rank(pct=True).mul(100).round(1)
    else:
        latest['linear_rank'] = 50.0

    latest = latest.sort_values('linear_rank', ascending=False)

    # ── Output ──
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    display_cols = [symbol_col, date_col, close_col, 'lin_slope', 'lin_r2', 'lin_pos',
                    'linear_strength', 'linear_rank', 'linear_flag', 'linear_label']
    display_cols = [c for c in display_cols if c in latest.columns]

    csv_path = outdir / f'{stem}.csv'
    latest[display_cols].to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")

    # ── Plotly scatter: R2 vs linear_rank ──
    html_path = None
    try:
        import plotly.express as px

        plot_df = latest.dropna(subset=['lin_r2', 'linear_rank']).copy()
        if not plot_df.empty:
            plot_df['flag_label'] = np.where(plot_df['linear_flag'], 'Linear Leader', 'Neutral')

            fig = px.scatter(
                plot_df,
                x='lin_r2',
                y='linear_rank',
                color='flag_label',
                color_discrete_map={'Linear Leader': '#00c853', 'Neutral': '#666666'},
                hover_name=symbol_col,
                hover_data={close_col: ':.2f', 'lin_slope': ':.4f', 'lin_pos': ':.2%'},
                size=plot_df['lin_pos'].clip(0, 1).fillna(0.01) * 30 + 5,
                title='Linear Trend Tracker',
                template='plotly_dark',
                labels={'lin_r2': 'R-squared (Trend Quality)', 'linear_rank': 'Linear Rank (percentile)'},
            )

            # Quadrant lines
            fig.add_vline(x=0.7, line_dash='dot', line_color='gray', opacity=0.4,
                          annotation_text='R2=0.7')
            fig.add_hline(y=70, line_dash='dot', line_color='gray', opacity=0.4)

            # Quadrant labels
            fig.add_annotation(x=0.85, y=90, text="CLEAN UPTREND<br>(Stage 2)",
                               showarrow=False, font=dict(color='#00c853', size=10))
            fig.add_annotation(x=0.85, y=20, text="CLEAN DOWNTREND<br>(Stage 4)",
                               showarrow=False, font=dict(color='#ff1744', size=10))
            fig.add_annotation(x=0.3, y=55, text="CHOPPY<br>(Stage 1/3)",
                               showarrow=False, font=dict(color='#ffd93d', size=10))

            fig.update_layout(
                font=dict(family='Consolas', size=12),
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                xaxis=dict(range=[-0.05, 1.05]),
                yaxis=dict(range=[-5, 105]),
            )

            html_path = outdir / f'{stem}_chart.html'
            fig.write_html(str(html_path))
            logger.info(f"Chart saved: {html_path}")

    except ImportError:
        logger.warning("Plotly not installed - skipping chart")

    return full, latest, csv_path, html_path


def print_linear_tracker_table(latest, symbol_col='symbol'):
    """Print the linear tracker leaderboard."""
    print(f"\n{'='*100}")
    print(f"  LINEAR TREND TRACKER - {len(latest)} tickers")
    print(f"{'='*100}")
    print(f"  {'Ticker':7s} {'Rank':>6s} {'Slope':>8s} {'R2':>6s} {'Return':>8s} {'Strength':>10s} {'Flag':>6s} {'Label':>16s}")
    print(f"  {'-'*75}")

    for _, row in latest.iterrows():
        sym = row.get(symbol_col, '?')
        rank = row.get('linear_rank', 0)
        slope = row.get('lin_slope', 0)
        r2 = row.get('lin_r2', 0)
        pos = row.get('lin_pos', 0)
        strength = row.get('linear_strength', 0)
        flag = '>>>' if row.get('linear_flag') else ''
        label = row.get('linear_label', '')

        if pd.isna(slope):
            slope = 0
        if pd.isna(r2):
            r2 = 0
        if pd.isna(pos):
            pos = 0
        if pd.isna(strength):
            strength = 0

        print(f"  {sym:7s} {rank:5.0f}% {slope:>+7.3f} {r2:5.3f} {pos:>+7.1%} {strength:>9.1f} {flag:>6s} {label:>16s}")

    print(f"{'='*100}")

    # Summary
    flagged = latest[latest.get('linear_flag', pd.Series(False)).fillna(False)]
    high_r2 = latest[latest['lin_r2'].fillna(0) > 0.7]
    print(f"\n  Linear Leaders (flagged): {len(flagged)}")
    for _, r in flagged.iterrows():
        print(f"    {r.get(symbol_col, '?'):7s}  slope={r.get('lin_slope', 0):+.3f}  "
              f"R2={r.get('lin_r2', 0):.3f}  return={r.get('lin_pos', 0):+.1%}")

    print(f"\n  High R2 (>0.7, any direction): {len(high_r2)}")
    for _, r in high_r2.iterrows():
        direction = 'UP' if (r.get('lin_slope', 0) or 0) > 0 else 'DOWN'
        print(f"    {r.get(symbol_col, '?'):7s}  {direction:4s}  slope={r.get('lin_slope', 0):+.3f}  "
              f"R2={r.get('lin_r2', 0):.3f}  return={r.get('lin_pos', 0):+.1%}")


def run_linear_tracker_mode(ohlcv_df, outdir='.', stamp=None, lookback=126):
    """Full linear tracker pipeline for CLI mode."""
    if stamp is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "=" * 60)
    print(f"  LINEAR TREND TRACKER (lookback={lookback} days)")
    print("=" * 60)

    full, latest, csv_path, html_path = linear_tracker(
        ohlcv_df, lookback=lookback, outdir=outdir, stem=f'linear_tracker_{stamp}'
    )

    print_linear_tracker_table(latest)

    print(f"\n  Outputs:")
    print(f"    CSV: {csv_path}")
    if html_path:
        print(f"    Chart: {html_path}")

    return full, latest, csv_path, html_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    print("Patch 18: Testing linear tracker...")
    df = pd.read_csv('ohlcv.csv')

    full, latest, csv_path, html_path = run_linear_tracker_mode(df)
    print(f"\nTimeseries rows: {len(full)}")
