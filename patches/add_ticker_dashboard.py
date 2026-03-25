"""
Patch 20: Single-Ticker Chart Dashboard
Generates a comprehensive one-pager for any ticker combining:
  - Price chart with 20/50/150-day SMAs
  - Volume bars with 50d average overlay
  - RS line vs SPY
  - Trend quality checklist (linear tracker attributes)
  - Scorecard table with all ranking/regime metrics
  - Event history timeline

CLI: --mode ticker_dashboard --symbol NVDA

Scorecard includes:
  absolute_strength_rank, comparative_strength_bucket, sector_rank_bucket,
  tradeability_score, linear_regime, iv_percentile, plus Weinstein stage,
  options structure, and all AS monitor flags.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from patches.add_linear_tracker import linear_tracker
from patches.add_linear_regime import linear_regime_alerts
from patches.add_absolute_strength import compute_absolute_strength_universe
from patches.add_comparative_strength import compute_comparative_strength, _get_sector
from patches.add_as_monitor import absolute_strength_monitor
from patches.options_overlay import options_overlay_for_ticker, compute_hv
from patches.iv_provider import fetch_iv_single

logger = logging.getLogger("ticker_dashboard")


def generate_ticker_dashboard(ohlcv_df, symbol, benchmark='SPY',
                               symbol_col='symbol', date_col='date',
                               close_col='close', open_col='open',
                               high_col='high', low_col='low',
                               volume_col='volume', outdir='.'):
    """
    Generate a full single-ticker dashboard as interactive HTML.
    """
    sym = symbol.upper()
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df[date_col] = pd.to_datetime(ohlcv_df[date_col])
    ohlcv_df = ohlcv_df.sort_values([symbol_col, date_col])

    ticker_df = ohlcv_df[ohlcv_df[symbol_col] == sym].copy().reset_index(drop=True)
    if ticker_df.empty:
        print(f"No data for {sym}")
        return None

    bench_df = ohlcv_df[ohlcv_df[symbol_col] == benchmark].copy().reset_index(drop=True)
    close = ticker_df[close_col].values.astype(float)
    dates = ticker_df[date_col].values
    volume = ticker_df[volume_col].values.astype(float)
    n = len(close)
    last = float(close[-1])
    last_date = str(dates[-1])[:10]

    # ── SMAs ──
    sma_20 = pd.Series(close).rolling(20).mean().values
    sma_50 = pd.Series(close).rolling(50).mean().values
    sma_150 = pd.Series(close).rolling(150).mean().values
    vol_50 = pd.Series(volume).rolling(50).mean().values

    # ── RS line vs benchmark ──
    rs_line = None
    if not bench_df.empty:
        bench_close = bench_df[close_col].values.astype(float)
        min_len = min(n, len(bench_close))
        if min_len > 20:
            s = close[-min_len:]
            b = bench_close[-min_len:]
            b_safe = np.where(b == 0, np.nan, b)
            rs_line = s / b_safe

    # ── Compute all metrics ──
    # Linear tracker
    full_lt, latest_lt, _, _ = linear_tracker(ohlcv_df, lookback=126, outdir=outdir, stem='_td_lt')
    lt_row = latest_lt[latest_lt[symbol_col] == sym]
    lt_data = lt_row.iloc[0].to_dict() if not lt_row.empty else {}

    # Linear regime
    full_lr = linear_regime_alerts(full_lt)
    lr_latest = full_lr.sort_values(date_col).groupby(symbol_col).tail(1)
    lr_row = lr_latest[lr_latest[symbol_col] == sym]
    lr_data = lr_row.iloc[0].to_dict() if not lr_row.empty else {}

    # Absolute strength
    abs_results = compute_absolute_strength_universe(ohlcv_df)
    abs_data = abs_results.get(sym, {})

    # Comparative strength
    comp_results, sector_results = compute_comparative_strength(ohlcv_df)
    comp_data = comp_results.get(sym, {})
    sector = comp_data.get('sector', _get_sector(sym))
    sector_info = sector_results.get(sector, {})

    # AS monitor
    as_monitor = absolute_strength_monitor(ohlcv_df)
    as_data = as_monitor.get(sym, {})

    # IV data
    iv_data = fetch_iv_single(sym, last)
    hv_20 = compute_hv(close, 20)
    hv_60 = compute_hv(close, 60)

    # Options overlay
    bench_close_arr = bench_df[close_col].values.astype(float) if not bench_df.empty else None
    from patches.add_stage_enriched import classify_universe
    stages = classify_universe(ohlcv_df, bench_df if not bench_df.empty else None)
    stage_data = stages.get(sym, {})

    overlay = options_overlay_for_ticker(
        close, stage=stage_data.get('stage'),
        pct_above_30w=stage_data.get('pct_above_30w'),
        slope_30w=stage_data.get('slope_30w'),
        real_iv=iv_data
    )

    # ── Build scorecard ──
    def _fmt(val, decimals=1, suffix='', default='n/a'):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return f"{round(val, decimals)}{suffix}"
        except Exception:
            return str(val)

    def _fmtn(val, decimals=1, suffix='', default='n/a'):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return f"{round(val, decimals)}{suffix}"

    def _flag(val):
        if val is True:
            return 'YES'
        if val is False:
            return '.'
        return 'n/a'

    scorecard = {
        'Weinstein Stage': stage_data.get('stage_label', 'n/a'),
        'Stage Confidence': _fmtn(stage_data.get('confidence'), 3),
        'Abs Strength Score': _fmtn(abs_data.get('abs_strength_score'), 1),
        'Abs Strength Label': abs_data.get('abs_strength_label', 'n/a'),
        'Abs Strength Rank': _fmtn(abs_data.get('abs_rank_20d'), 0, 'th pctile'),
        'Perf 20d': _fmtn(abs_data.get('abs_perf_20d'), 1, '%'),
        'RS vs SPY 20d': _fmtn(abs_data.get('rs_vs_spy_20d'), 1, '%'),
        'Mansfield RS': _fmtn(abs_data.get('mansfield_rs_52w'), 1),
        'Sector': sector,
        'Sector Rank': f"#{comp_data.get('sector_rank', '?')} of {comp_data.get('sector_size', '?')}",
        'Sector Rank Pctile': _fmtn(comp_data.get('sector_rank_pct'), 0, 'th'),
        'Sector Rotation': sector_info.get('rotation_signal', 'n/a'),
        'RS Regime': comp_data.get('relative_strength_regime', 'n/a'),
        'Tradeability Score': _fmtn(overlay.get('tradeability_score', as_data.get('as_composite_score')), 1, '/20'),
        'HV 20d': _fmtn(hv_20, 1, '%'),
        'HV 60d': _fmtn(hv_60, 1, '%'),
        'ATM IV': _fmtn(iv_data.get('atm_iv'), 1, '%') if iv_data.get('source') == 'options_chain' else 'n/a (HV proxy)',
        'IV Percentile': _fmtn(overlay.get('iv_percentile'), 0, 'th'),
        'IV Regime': overlay.get('iv_regime', 'n/a'),
        'Options Structure': overlay.get('structure', 'n/a'),
        'DTE Range': overlay.get('dte_range', 'n/a'),
        'Linear Slope': _fmtn(lt_data.get('lin_slope'), 3),
        'Linear R2': _fmtn(lt_data.get('lin_r2'), 3),
        'Linear Position': _fmtn(lt_data.get('lin_pos'), 1, '%') if lt_data.get('lin_pos') else 'n/a',
        'Linear Regime': lr_data.get('linear_regime', 'n/a'),
        'Linear Flag': _flag(lt_data.get('linear_flag')),
    }

    trend_checklist = {
        'Price > 20d SMA': _flag(as_data.get('above_20sma')),
        'Price > 50d SMA': _flag(as_data.get('above_50sma')),
        'Golden Cross (20>50)': _flag(as_data.get('golden_cross_20_50')),
        'RS at 20d High': _flag(as_data.get('rs_at_20d_high')),
        'Near 52w High': _flag(as_data.get('near_52w_high')),
        'New 20d High': _flag(as_data.get('new_20d_high')),
        'Volume Confirmation': _flag(as_data.get('volume_confirmation')),
        'AS Composite': f"{as_data.get('as_composite_score', 0)}/7",
        'AS Flagged': _flag(as_data.get('as_flagged')),
    }

    # ── Build HTML ──
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # Scorecard rows
    scorecard_rows = ''.join(
        f'<tr><td style="text-align:left;padding:4px 12px;color:#8b949e;">{k}</td>'
        f'<td style="text-align:right;padding:4px 12px;font-weight:bold;">{v}</td></tr>'
        for k, v in scorecard.items()
    )

    checklist_rows = ''.join(
        f'<tr><td style="text-align:left;padding:3px 12px;color:#8b949e;">{k}</td>'
        f'<td style="text-align:right;padding:3px 12px;font-weight:bold;'
        f'{"color:#00c853" if v == "YES" else "color:#666" if v == "." else ""}">{v}</td></tr>'
        for k, v in trend_checklist.items()
    )

    # Charts via Plotly
    charts_html = ""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.2, 0.25],
            vertical_spacing=0.03,
            subplot_titles=(f'{sym} Price + SMAs', 'Volume', 'RS Line vs SPY')
        )

        # Price + SMAs
        fig.add_trace(go.Scatter(x=dates, y=close, name='Close', line=dict(color='#e0e0e0', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sma_20, name='20 SMA', line=dict(color='#58a6ff', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sma_50, name='50 SMA', line=dict(color='#ffd93d', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=sma_150, name='150 SMA (30w)', line=dict(color='#ff6b6b', width=1.5)), row=1, col=1)

        # Volume
        vol_colors = ['#00c853' if close[i] >= (close[i-1] if i > 0 else close[i]) else '#ff1744' for i in range(n)]
        fig.add_trace(go.Bar(x=dates, y=volume, name='Volume', marker_color=vol_colors, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=vol_50, name='50d Avg Vol', line=dict(color='#ffd93d', width=1)), row=2, col=1)

        # RS line
        if rs_line is not None:
            rs_dates = dates[-len(rs_line):]
            fig.add_trace(go.Scatter(x=rs_dates, y=rs_line, name='RS vs SPY', line=dict(color='#58a6ff', width=1.5)), row=3, col=1)
            rs_20_sma = pd.Series(rs_line).rolling(20).mean().values
            fig.add_trace(go.Scatter(x=rs_dates, y=rs_20_sma, name='RS 20d SMA', line=dict(color='#ffd93d', width=1, dash='dot')), row=3, col=1)

        fig.update_layout(
            template='plotly_dark',
            height=700,
            showlegend=True,
            legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center', font=dict(size=10)),
            font=dict(family='Consolas', size=11),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            margin=dict(l=50, r=20, t=40, b=30),
        )
        fig.update_yaxes(gridcolor='#1a1a2e')
        fig.update_xaxes(gridcolor='#1a1a2e')

        charts_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    except ImportError:
        charts_html = '<p style="color:#666;">Plotly not installed - charts unavailable</p>'

    # Stage color
    stage_num = stage_data.get('stage')
    stage_color = {1: '#ffd93d', 2: '#00c853', 3: '#ff9800', 4: '#ff1744'}.get(stage_num, '#666')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{sym} - Ticker Dashboard</title>
    <style>
        body {{ background:#0d1117; color:#e0e0e0; font-family:Consolas,monospace; padding:20px; margin:0; }}
        h1 {{ color:#58a6ff; margin-bottom:5px; }}
        .subtitle {{ color:#8b949e; font-size:13px; margin-bottom:15px; }}
        .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:15px; }}
        .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }}
        .card h3 {{ color:#58a6ff; margin:0 0 10px 0; font-size:14px; border-bottom:1px solid #30363d; padding-bottom:6px; }}
        table {{ width:100%; border-collapse:collapse; font-size:12px; }}
        .stage-badge {{ display:inline-block; padding:4px 12px; border-radius:4px; font-weight:bold;
                        background:{stage_color}; color:{'white' if stage_num in (2,4) else 'black'}; }}
        .header-stats {{ display:flex; gap:16px; margin:10px 0; flex-wrap:wrap; }}
        .stat {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:8px 14px; }}
        .stat .label {{ color:#8b949e; font-size:10px; text-transform:uppercase; }}
        .stat .value {{ color:#58a6ff; font-size:18px; font-weight:bold; }}
    </style>
</head>
<body>
    <h1>{sym} <span class="stage-badge">{stage_data.get('stage_label', 'Unknown')}</span></h1>
    <div class="subtitle">${last:.2f} | {last_date} | {sector} | {comp_data.get('relative_strength_regime', '')}</div>

    <div class="header-stats">
        <div class="stat"><div class="label">AS Score</div><div class="value">{abs_data.get('abs_strength_score', 'n/a')}</div></div>
        <div class="stat"><div class="label">AS Rank</div><div class="value">{_fmtn(abs_data.get('abs_rank_20d'), 0)}th</div></div>
        <div class="stat"><div class="label">Sector Rank</div><div class="value">#{comp_data.get('sector_rank', '?')}</div></div>
        <div class="stat"><div class="label">IV Regime</div><div class="value">{overlay.get('iv_regime', 'n/a')}</div></div>
        <div class="stat"><div class="label">Linear R2</div><div class="value">{_fmtn(lt_data.get('lin_r2'), 3)}</div></div>
        <div class="stat"><div class="label">AS Monitor</div><div class="value">{as_data.get('as_composite_score', 0)}/7</div></div>
    </div>

    {charts_html}

    <div class="grid">
        <div class="card">
            <h3>Trend Quality Checklist</h3>
            <table>{checklist_rows}</table>
        </div>
        <div class="card">
            <h3>Full Scorecard</h3>
            <table>{scorecard_rows}</table>
        </div>
    </div>

    <p style="color:#444;font-size:10px;margin-top:20px;">
        APEX Trade Analyzer | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
</body>
</html>"""

    html_path = outdir / f'ticker_dashboard_{sym}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Ticker dashboard saved: {html_path}")
    return html_path


def run_ticker_dashboard_mode(ohlcv_df, symbol, outdir='.'):
    """CLI entry point for --mode ticker_dashboard."""
    print(f"\n{'='*60}")
    print(f"  TICKER DASHBOARD — {symbol.upper()}")
    print(f"{'='*60}")

    path = generate_ticker_dashboard(ohlcv_df, symbol, outdir=outdir)
    return path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', '-s', default='NVDA')
    parser.add_argument('--input', '-i', default='ohlcv.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    run_ticker_dashboard_mode(df, args.symbol)
