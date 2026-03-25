"""
Patch 17: Master Dashboard
Generates styled HTML + CSV dashboard from master_matrix output.

write_master_dashboard():
  - Top 50 by master_matrix_score
  - RdYlGn heatmaps on all score/rank columns
  - Color-coded tiers, sell signal highlights
  - Bonus detail column
  - Dark-theme HTML with summary stat cards
  - Auto-generates when running --mode master_matrix

Also adds --mode master_dashboard (standalone from existing CSV).

CLI:
  python patches/add_cli_runner.py -i ohlcv.csv -m master_matrix --dashboard
  python patches/add_cli_runner.py -i master_matrix_full.csv -m master_dashboard
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("master_dashboard")


def write_master_dashboard(df, outdir='.', stem='master_dashboard', top_n=50):
    """
    Generate top-N master dashboard as CSV + styled HTML.

    Applies RdYlGn heatmaps to all score/rank columns, color-codes
    tiers, highlights sell signals, and wraps in a dark-theme HTML page.
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    # ── Select and sort ──
    score_col = 'master_matrix_score' if 'master_matrix_score' in df.columns else 'master_score'
    if score_col not in df.columns:
        logger.warning(f"No {score_col} column — using event_score")
        score_col = 'event_score'

    events = df.copy()
    if 'event_code' in events.columns:
        events = events[events['event_code'] != 'none']

    # HARD FILTER: only tradeable signals
    if 'tradeability_flag' in events.columns:
        events = events[events['tradeability_flag'].fillna(False)]

    events = events.nlargest(top_n, score_col)

    # ── Select display columns ──
    display_cols = [c for c in [
        'symbol', 'date', 'close',
        'master_matrix_score', 'mm_tier', 'mm_raw_score',
        'event_label', 'event_score',
        'w_stage_label',
        'tradeability_score',
        'abs_strength_score', 'abs_rank_20d', 'abs_strength_label',
        'sector', 'sector_rank_pct', 'sector_rotation_signal',
        'iv_regime', 'atm_iv', 'hv_20', 'structure',
        'climax_penalty', 'fill_risk_penalty',
        'mm_bonus', 'mm_bonus_detail',
        'sell_signal',
        'mm_breakdown',
    ] if c in events.columns]

    top = events[display_cols].copy()

    # Format columns
    if 'date' in top.columns:
        top['date'] = pd.to_datetime(top['date']).dt.strftime('%Y-%m-%d')
    for col in ['close', 'atm_iv', 'hv_20']:
        if col in top.columns:
            top[col] = pd.to_numeric(top[col], errors='coerce').round(2)

    # ── CSV ──
    csv_path = outdir / f'{stem}_top{top_n}.csv'
    top.to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")

    # ── Styled HTML ──
    html_path = outdir / f'{stem}_top{top_n}.html'

    styler = top.style

    # RdYlGn heatmaps on score columns
    heatmap_cols = [c for c in [
        'master_matrix_score', 'mm_raw_score', 'event_score',
        'tradeability_score', 'abs_strength_score', 'abs_rank_20d',
        'sector_rank_pct', 'mm_bonus',
    ] if c in top.columns]

    for col in heatmap_cols:
        try:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn')
        except Exception:
            pass

    # Reverse heatmap on penalty columns (high = bad = red)
    penalty_cols = [c for c in ['climax_penalty', 'fill_risk_penalty'] if c in top.columns]
    for col in penalty_cols:
        try:
            styler = styler.background_gradient(subset=[col], cmap='RdYlGn_r')
        except Exception:
            pass

    # Color-code tiers
    if 'mm_tier' in top.columns:
        tier_colors = {
            'Elite': 'background-color: #00c853; color: white; font-weight: bold',
            'Very Strong': 'background-color: #4caf50; color: white',
            'Strong': 'background-color: #8bc34a; color: black',
            'Average': 'background-color: #ffeb3b; color: black',
            'Below Avg': 'background-color: #ff9800; color: black',
            'Weak': 'background-color: #f44336; color: white',
            'Very Weak': 'background-color: #b71c1c; color: white; font-weight: bold',
        }
        styler = styler.map(
            lambda v: tier_colors.get(v, ''),
            subset=['mm_tier']
        )

    # Highlight sell signals
    if 'sell_signal' in top.columns:
        styler = styler.map(
            lambda v: 'background-color: #ff4444; color: white; font-weight: bold'
            if v is True else '',
            subset=['sell_signal']
        )

    # IV regime colors
    if 'iv_regime' in top.columns:
        iv_colors = {
            'rich': 'background-color: #ff6b6b; color: white',
            'fair': 'background-color: #ffd93d',
            'cheap': 'background-color: #6bcb77; color: white',
        }
        styler = styler.map(
            lambda v: iv_colors.get(v, ''),
            subset=['iv_regime']
        )

    # Rotation signal colors
    if 'sector_rotation_signal' in top.columns:
        rot_colors = {
            'into': 'background-color: #00c853; color: white',
            'accelerating': 'background-color: #8bc34a',
            'decelerating': 'background-color: #ff9800',
            'out': 'background-color: #f44336; color: white',
        }
        styler = styler.map(
            lambda v: rot_colors.get(str(v), ''),
            subset=['sector_rotation_signal']
        )

    styler = styler.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1a1a2e'), ('color', '#e0e0e0'),
            ('padding', '6px 8px'), ('font-size', '11px'), ('text-align', 'center'),
            ('position', 'sticky'), ('top', '0'), ('z-index', '1'),
        ]},
        {'selector': 'td', 'props': [
            ('padding', '4px 8px'), ('font-size', '11px'), ('text-align', 'center'),
            ('border-bottom', '1px solid #333'), ('white-space', 'nowrap'),
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'), ('font-family', 'Consolas, monospace'),
            ('width', '100%'),
        ]},
    ])

    styled_html = styler.to_html()

    # Summary stats
    n_elite = (top.get('mm_tier', pd.Series()) == 'Elite').sum()
    n_sells = top.get('sell_signal', pd.Series(False)).sum()
    avg_score = top[score_col].mean() if score_col in top.columns else 0
    top_sym = top.iloc[0]['symbol'] if len(top) > 0 and 'symbol' in top.columns else '?'
    top_score_val = top.iloc[0][score_col] if len(top) > 0 and score_col in top.columns else 0

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>APEX Master Dashboard</title>
    <style>
        body {{
            background-color: #0d1117;
            color: #e0e0e0;
            font-family: Consolas, 'Courier New', monospace;
            padding: 20px;
            margin: 0;
        }}
        h1 {{
            color: #58a6ff;
            border-bottom: 2px solid #30363d;
            padding-bottom: 10px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #8b949e;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 18px;
            min-width: 120px;
        }}
        .stat-card .label {{ color: #8b949e; font-size: 11px; text-transform: uppercase; }}
        .stat-card .value {{ color: #58a6ff; font-size: 22px; font-weight: bold; }}
        .stat-card.elite .value {{ color: #00c853; }}
        .stat-card.sell .value {{ color: #ff4444; }}
        .table-container {{
            overflow-x: auto;
            margin-top: 10px;
        }}
        .legend {{
            display: flex;
            gap: 12px;
            margin: 15px 0;
            flex-wrap: wrap;
            font-size: 11px;
        }}
        .legend span {{
            padding: 3px 8px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>APEX Master Matrix Dashboard</h1>
    <div class="subtitle">
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Top {top_n} signals by master_matrix_score
    </div>
    <div class="stats">
        <div class="stat-card">
            <div class="label">Signals</div>
            <div class="value">{len(top)}</div>
        </div>
        <div class="stat-card elite">
            <div class="label">Elite Tier</div>
            <div class="value">{n_elite}</div>
        </div>
        <div class="stat-card sell">
            <div class="label">Sell Signals</div>
            <div class="value">{n_sells}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Score</div>
            <div class="value">{avg_score:.1f}</div>
        </div>
        <div class="stat-card elite">
            <div class="label">#1 Signal</div>
            <div class="value">{top_sym} ({top_score_val:.1f})</div>
        </div>
    </div>
    <div class="legend">
        <span style="background:#00c853;color:white;">Elite</span>
        <span style="background:#4caf50;color:white;">Very Strong</span>
        <span style="background:#8bc34a;">Strong</span>
        <span style="background:#ffeb3b;">Average</span>
        <span style="background:#ff9800;">Below Avg</span>
        <span style="background:#f44336;color:white;">Weak</span>
        <span style="background:#b71c1c;color:white;">Very Weak</span>
    </div>
    <div class="table-container">
        {styled_html}
    </div>
    <p style="color:#555;font-size:10px;margin-top:20px;">
        APEX Trade Analyzer Suite | master_matrix_score = event + tradeability + strength + sector + bonuses - penalties
    </p>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    logger.info(f"HTML saved: {html_path}")

    print(f"  Dashboard CSV: {csv_path}")
    print(f"  Dashboard HTML: {html_path}")

    return csv_path, html_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Try to load existing master matrix output
    candidates = sorted(Path('.').glob('master_matrix_*.csv'), reverse=True)
    if candidates:
        print(f"Loading {candidates[0]}...")
        df = pd.read_csv(candidates[0])
        write_master_dashboard(df)
    else:
        print("No master_matrix CSV found. Run --mode master_matrix first.")
