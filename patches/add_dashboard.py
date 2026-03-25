"""
Patch 6: Dashboard Export + Plotly Visualizations
Adds: write_dashboard() and write_plots(), hooked into main_cli()

write_dashboard() — exports top 20 by event_score as CSV + styled HTML with RdYlGn heatmaps
write_plots()     — Plotly gap histogram + gap_fill_risk vs score scatter colored by event_label
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime


def write_dashboard(df, output_prefix='dashboard', top_n=20,
                    score_col='event_score', label_col='event_label'):
    """
    Export top N events as CSV + styled HTML with RdYlGn heatmaps.

    Outputs:
      - {output_prefix}_top{top_n}.csv
      - {output_prefix}_top{top_n}.html
    """
    # Select display columns
    display_cols = [c for c in [
        'symbol', 'date', 'close', 'gap_pct', 'signal', 'direction',
        score_col, label_col, 'gap_fill_risk', 'fill_distance_pct',
        'filled_same_day', 'is_earnings_gap', 'climax_top', 'sell_signal',
        'bull_score', 'bear_score',
    ] if c in df.columns]

    # Filter to events only and take top N
    events = df.copy()
    if 'event_code' in events.columns:
        events = events[events['event_code'] != 'none']
    if score_col in events.columns:
        events = events.nlargest(top_n, score_col)
    else:
        events = events.head(top_n)

    top = events[display_cols].copy()

    # Format columns for display
    if 'gap_pct' in top.columns:
        top['gap_pct'] = (top['gap_pct'] * 100).round(2).astype(str) + '%'
    if 'fill_distance_pct' in top.columns:
        top['fill_distance_pct'] = top['fill_distance_pct'].round(3)
    if 'date' in top.columns:
        top['date'] = pd.to_datetime(top['date']).dt.strftime('%Y-%m-%d')

    # ── CSV export ──
    csv_path = f"{output_prefix}_top{top_n}.csv"
    top.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ── Styled HTML export ──
    html_path = f"{output_prefix}_top{top_n}.html"

    # Build styled HTML
    styler = top.style

    # RdYlGn heatmap on score column (green = high conviction)
    if score_col in top.columns:
        styler = styler.background_gradient(
            subset=[score_col], cmap='RdYlGn', vmin=0, vmax=10
        )

    # RdYlGn on bull_score / bear_score
    for col in ['bull_score', 'bear_score']:
        if col in top.columns:
            styler = styler.background_gradient(
                subset=[col], cmap='RdYlGn', vmin=0, vmax=7
            )

    # Red/green on fill_distance_pct (red = high fill = bad for gap holders)
    if 'fill_distance_pct' in top.columns:
        styler = styler.background_gradient(
            subset=['fill_distance_pct'], cmap='RdYlGn_r', vmin=0, vmax=1.5
        )

    # Highlight sell signals in red
    if 'sell_signal' in top.columns:
        styler = styler.map(
            lambda v: 'background-color: #ff6b6b; color: white; font-weight: bold'
            if v is True else '',
            subset=['sell_signal']
        )

    # Highlight climax tops
    if 'climax_top' in top.columns:
        styler = styler.map(
            lambda v: 'background-color: #ff4444; color: white; font-weight: bold'
            if v is True else '',
            subset=['climax_top']
        )

    # Highlight earnings gaps
    if 'is_earnings_gap' in top.columns:
        styler = styler.map(
            lambda v: 'background-color: #ffd93d; font-weight: bold'
            if v is True else '',
            subset=['is_earnings_gap']
        )

    # Color gap_fill_risk
    if 'gap_fill_risk' in top.columns:
        risk_colors = {
            'high': 'background-color: #ff6b6b; color: white',
            'medium': 'background-color: #ffd93d',
            'low': 'background-color: #6bcb77; color: white',
        }
        styler = styler.map(
            lambda v: risk_colors.get(v, ''),
            subset=['gap_fill_risk']
        )

    styler = styler.set_caption(
        f"Trade Event Dashboard - Top {top_n} by Score - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    styler = styler.set_table_styles([
        {'selector': 'caption', 'props': [
            ('font-size', '18px'), ('font-weight', 'bold'), ('margin-bottom', '10px')
        ]},
        {'selector': 'th', 'props': [
            ('background-color', '#1a1a2e'), ('color', '#e0e0e0'),
            ('padding', '8px 12px'), ('font-size', '12px'), ('text-align', 'center')
        ]},
        {'selector': 'td', 'props': [
            ('padding', '6px 10px'), ('font-size', '12px'), ('text-align', 'center'),
            ('border-bottom', '1px solid #333')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'), ('font-family', 'Consolas, monospace'),
            ('width', '100%')
        ]},
    ])

    # Wrap in a full HTML page with dark theme
    styled_html = styler.to_html()
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Trade Event Dashboard</title>
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
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px 20px;
            min-width: 140px;
        }}
        .stat-card .label {{ color: #8b949e; font-size: 12px; }}
        .stat-card .value {{ color: #58a6ff; font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Trade Event Dashboard</h1>
    <div class="stats">
        <div class="stat-card">
            <div class="label">Total Events</div>
            <div class="value">{len(top)}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Score</div>
            <div class="value">{top[score_col].mean():.1f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Sell Signals</div>
            <div class="value">{top['sell_signal'].sum() if 'sell_signal' in top.columns else 0}</div>
        </div>
    </div>
    {styled_html}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"HTML saved: {html_path}")

    return csv_path, html_path


def write_plots(df, output_prefix='dashboard',
                score_col='event_score', label_col='event_label'):
    """
    Generate Plotly visualizations:
      1. Gap histogram — distribution of gap sizes by direction
      2. Gap fill risk vs score scatter — colored by event label

    Outputs:
      - {output_prefix}_gap_histogram.html
      - {output_prefix}_scatter.html
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        print("Skipping plot generation.")
        return None, None

    # Filter to gap days only
    gaps = df.copy()
    if 'gap_up' in gaps.columns and 'gap_down' in gaps.columns:
        gaps = gaps[gaps['gap_up'].fillna(False) | gaps['gap_down'].fillna(False)]

    if len(gaps) == 0:
        print("No gap data to plot.")
        return None, None

    # ── 1. Gap Histogram ──
    hist_path = f"{output_prefix}_gap_histogram.html"

    if 'gap_pct' in gaps.columns and 'direction' in gaps.columns:
        gaps['gap_pct_display'] = gaps['gap_pct'] * 100

        fig1 = px.histogram(
            gaps,
            x='gap_pct_display',
            color='direction',
            nbins=50,
            color_discrete_map={'bullish': '#00c853', 'bearish': '#ff1744'},
            labels={'gap_pct_display': 'Gap Size (%)', 'direction': 'Direction'},
            title='Gap Size Distribution by Direction',
            template='plotly_dark',
            opacity=0.75,
        )
        fig1.update_layout(
            barmode='overlay',
            xaxis_title='Gap Size (%)',
            yaxis_title='Count',
            font=dict(family='Consolas', size=12),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
        )
        fig1.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.4)
        fig1.write_html(hist_path)
        print(f"Gap histogram saved: {hist_path}")
    else:
        print("Missing gap_pct or direction columns - skipping histogram")
        hist_path = None

    # ── 2. Gap Fill Risk vs Score Scatter ──
    scatter_path = f"{output_prefix}_scatter.html"

    has_fill = 'fill_distance_pct' in gaps.columns
    has_score = score_col in gaps.columns
    has_label = label_col in gaps.columns

    if has_fill and has_score:
        plot_df = gaps.dropna(subset=['fill_distance_pct', score_col]).copy()

        if len(plot_df) == 0:
            print("No data with both fill_distance_pct and score - skipping scatter")
            scatter_path = None
        else:
            # Cap fill_distance for display
            plot_df['fill_dist_capped'] = plot_df['fill_distance_pct'].clip(0, 2.0)

            color_col = label_col if has_label else 'direction'
            hover_data = {
                'symbol': True,
                'gap_pct': ':.2%',
                'gap_fill_risk': True,
            }

            fig2 = px.scatter(
                plot_df,
                x='fill_dist_capped',
                y=score_col,
                color=color_col if color_col in plot_df.columns else None,
                size=plot_df['gap_pct'].abs() * 500 + 5,
                hover_name='symbol' if 'symbol' in plot_df.columns else None,
                hover_data={k: v for k, v in hover_data.items() if k in plot_df.columns},
                labels={
                    'fill_dist_capped': 'Gap Fill Distance (0=held, 1=filled)',
                    score_col: 'Event Score',
                },
                title='Gap Fill Risk vs Event Score',
                template='plotly_dark',
                opacity=0.7,
            )

            # Add quadrant lines
            fig2.add_hline(y=5, line_dash='dot', line_color='gray', opacity=0.4)
            fig2.add_vline(x=0.5, line_dash='dot', line_color='gray', opacity=0.4)

            # Annotate quadrants
            fig2.add_annotation(x=0.15, y=8.5, text="HIGH CONVICTION<br>Gap Held",
                                showarrow=False, font=dict(color='#00c853', size=10))
            fig2.add_annotation(x=1.5, y=8.5, text="HIGH SCORE<br>But Gap Filled",
                                showarrow=False, font=dict(color='#ffd93d', size=10))
            fig2.add_annotation(x=0.15, y=1.5, text="LOW CONVICTION<br>Gap Held",
                                showarrow=False, font=dict(color='gray', size=10))
            fig2.add_annotation(x=1.5, y=1.5, text="WEAK<br>Filled + Low Score",
                                showarrow=False, font=dict(color='#ff1744', size=10))

            fig2.update_layout(
                font=dict(family='Consolas', size=12),
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                xaxis=dict(range=[-0.05, 2.1]),
                yaxis=dict(range=[-0.5, 10.5]),
            )
            fig2.write_html(scatter_path)
            print(f"Scatter plot saved: {scatter_path}")
    else:
        print("Missing fill_distance_pct or score columns - skipping scatter")
        scatter_path = None

    return hist_path, scatter_path


if __name__ == '__main__':
    from breakaway_gap_scan import breakaway_gap_scan
    from patches.add_gap_fill_risk import add_gap_fill_risk
    from patches.add_post_earnings_flag import post_earnings_flag_scan
    from patches.add_composite_event import add_composite_event
    from patches.add_unified_scanner import unified_event_scan

    print("Patch 6: Generating dashboard + plots...")
    df = pd.read_csv('ohlcv.csv')
    result = unified_event_scan(df)

    write_dashboard(result, output_prefix='dashboard', top_n=20)
    write_plots(result, output_prefix='dashboard')
    print("\nDone.")
