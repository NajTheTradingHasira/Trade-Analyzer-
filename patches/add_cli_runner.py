"""
Patch 5: CLI Runner
Adds: main_cli() + argparse

Unified command-line interface for all scanner modes:
  python patches/add_cli_runner.py --input ohlcv.csv --mode unified
  python patches/add_cli_runner.py --input ohlcv.csv --mode breakaway
  python patches/add_cli_runner.py --input ohlcv.csv --mode climax
  python patches/add_cli_runner.py --input ohlcv.csv --mode gaps-only --min-score 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime

from breakaway_gap_scan import breakaway_gap_scan
from patches.add_gap_fill_risk import add_gap_fill_risk
from patches.add_post_earnings_flag import post_earnings_flag_scan
from patches.add_composite_event import add_composite_event
from patches.add_unified_scanner import climax_top_scan, unified_event_scan
from patches.add_stage_enriched import stage_enriched_scan
from patches.add_backtester import backtest_events
from patches.add_as_monitor import run_absolute_strength_mode
from patches.add_theme_momentum import run_theme_momentum_mode
from patches.add_master_score import run_master_mode
from patches.add_master_matrix import run_master_matrix
from patches.add_master_dashboard import write_master_dashboard
from patches.add_linear_tracker import run_linear_tracker_mode
from patches.add_linear_regime import run_linear_regime_mode
from patches.add_ticker_dashboard import run_ticker_dashboard_mode
from patches.add_dashboard import write_dashboard, write_plots


BANNER = """
=== TRADE EVENT SCANNER - CLI ===
Modes: unified | breakaway | climax | gaps-only
"""


def main_cli():
    parser = argparse.ArgumentParser(
        description="Trade Event Scanner — detect breakaway gaps, climax tops, and composite events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python patches/add_cli_runner.py --input ohlcv.csv --mode unified
  python patches/add_cli_runner.py --input ohlcv.csv --mode climax --output climax_alerts.csv
  python patches/add_cli_runner.py --input ohlcv.csv --mode breakaway --min-score 4.0 --top 20
  python patches/add_cli_runner.py --input ohlcv.csv --mode gaps-only --symbol AAPL
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input CSV file with OHLCV data')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file (default: auto-named)')
    parser.add_argument('--mode', '-m', default='unified',
                        choices=['unified', 'breakaway', 'climax', 'gaps-only', 'stage_enriched', 'backtest', 'absolute_strength', 'theme_momentum', 'master', 'master_matrix', 'master_dashboard', 'linear_tracker', 'linear_regime', 'ticker_dashboard'],
                        help='Scanner mode (default: unified)')
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='Minimum event score to include (default: 0)')
    parser.add_argument('--top', '-n', type=int, default=50,
                        help='Show top N results (default: 50)')
    parser.add_argument('--symbol', '-s', default=None,
                        help='Filter to a single symbol')
    parser.add_argument('--events-only', action='store_true',
                        help='Only output rows with detected events')
    parser.add_argument('--date-from', default=None,
                        help='Filter: start date (YYYY-MM-DD)')
    parser.add_argument('--date-to', default=None,
                        help='Filter: end date (YYYY-MM-DD)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--dashboard', action='store_true',
                        help='Export top 20 as styled HTML dashboard + CSV')
    parser.add_argument('--plots', action='store_true',
                        help='Generate Plotly gap histogram + scatter plot')
    parser.add_argument('--dashboard-top', type=int, default=20,
                        help='Number of events in dashboard (default: 20)')

    # Column name overrides
    parser.add_argument('--col-symbol', default='symbol')
    parser.add_argument('--col-date', default='date')
    parser.add_argument('--col-open', default='open')
    parser.add_argument('--col-high', default='high')
    parser.add_argument('--col-low', default='low')
    parser.add_argument('--col-close', default='close')
    parser.add_argument('--col-volume', default='volume')

    args = parser.parse_args()

    if not args.quiet:
        print(BANNER)
        print(f"Input:  {args.input}")
        print(f"Mode:   {args.mode}")
        print(f"Filter: score >= {args.min_score}")
        if args.symbol:
            print(f"Symbol: {args.symbol}")
        print()

    # ── Load data ──
    df = pd.read_csv(args.input)
    if not args.quiet:
        print(f"Loaded {len(df)} rows, {df[args.col_symbol].nunique()} symbols")

    # Filter by symbol
    if args.symbol:
        df = df[df[args.col_symbol].str.upper() == args.symbol.upper()]
        if df.empty:
            print(f"No data found for symbol: {args.symbol}")
            return

    # Filter by date
    if args.date_from:
        df[args.col_date] = pd.to_datetime(df[args.col_date])
        df = df[df[args.col_date] >= args.date_from]
    if args.date_to:
        df[args.col_date] = pd.to_datetime(df[args.col_date])
        df = df[df[args.col_date] <= args.date_to]

    # ── Run scanner ──
    if args.mode == 'ticker_dashboard':
        if not args.symbol:
            print("ERROR: --symbol required for ticker_dashboard mode")
            return
        path = run_ticker_dashboard_mode(df, args.symbol, outdir='.')
        if path:
            result = pd.DataFrame({'file': [str(path)]})
        else:
            result = pd.DataFrame()
    elif args.mode == 'linear_regime':
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full, latest, csv = run_linear_regime_mode(df, outdir='.', stamp=stamp)
        result = latest
    elif args.mode == 'linear_tracker':
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full, latest, csv_path, html_path = run_linear_tracker_mode(df, outdir='.', stamp=stamp)
        result = latest
    elif args.mode == 'master_dashboard':
        # Standalone: load existing CSV and generate dashboard
        result = df  # Input is already the master_matrix CSV
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        write_master_dashboard(result, outdir='.', stem=f'master_dashboard_{stamp}')
    elif args.mode == 'master_matrix':
        result, regime, stages, crossovers = run_master_matrix(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
        # Save top 50 CSV
        events = result[result.get('event_code', pd.Series('none')) != 'none']
        mm_stem = args.output or f"master_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        top50_path = os.path.splitext(mm_stem)[0] + '_top50.csv'
        events.head(50).to_csv(top50_path, index=False)
        if not args.quiet:
            print(f"  Top 50 saved: {top50_path}")
        # Auto-generate master dashboard
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        write_master_dashboard(result, outdir='.', stem=f'master_dashboard_{stamp}')
    elif args.mode == 'master':
        result, regime, stages, sector_ts, crossovers = run_master_mode(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
    elif args.mode == 'theme_momentum':
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sector_ts, csv_path, html_path = run_theme_momentum_mode(df, outdir='.', stamp=stamp)
        # Build a minimal result for output compatibility
        result = sector_ts
    elif args.mode == 'absolute_strength':
        monitor_results, rating_alerts = run_absolute_strength_mode(
            df, args.col_symbol, args.col_date, args.col_close, args.col_volume
        )
        # Build a DataFrame from monitor results for output/dashboard
        rows = []
        for sym, d in monitor_results.items():
            rows.append({args.col_symbol: sym, **d})
        result = pd.DataFrame(rows).sort_values('as_composite_score', ascending=False)
        # Add alert tier info
        alert_map = {a['symbol']: a.get('tier_label', '') for a in rating_alerts}
        result['as_tier'] = result[args.col_symbol].map(alert_map).fillna('')
    elif args.mode == 'backtest':
        result, tables, regime, stages = backtest_events(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
        # Save backtest tables
        bt_prefix = os.path.splitext(output_file if args.output else f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")[0]
        for name, table in tables.items():
            if not table.empty:
                path = f"{bt_prefix}_{name}.csv"
                table.to_csv(path, index=False)
                if not args.quiet:
                    print(f"  Saved: {path}")
    elif args.mode == 'stage_enriched':
        result, regime, stages = stage_enriched_scan(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
    elif args.mode == 'unified':
        result = unified_event_scan(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
    elif args.mode == 'breakaway':
        result = breakaway_gap_scan(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
        result = add_gap_fill_risk(
            result, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
        result = add_composite_event(result, args.col_symbol, args.col_date)
    elif args.mode == 'climax':
        result = climax_top_scan(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )
    elif args.mode == 'gaps-only':
        result = breakaway_gap_scan(
            df, args.col_symbol, args.col_date, args.col_open,
            args.col_high, args.col_low, args.col_close, args.col_volume
        )

    # ── Filter results ──
    if args.events_only and 'event_code' in result.columns:
        result = result[result['event_code'] != 'none']
    if args.min_score > 0:
        score_col = 'event_score' if 'event_score' in result.columns else 'score'
        if score_col in result.columns:
            result = result[result[score_col] >= args.min_score]

    # ── Output ──
    output_file = args.output or f"{args.mode}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_csv(output_file, index=False)

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"RESULTS — {args.mode.upper()} SCAN")
        print(f"{'='*60}")
        print(f"Total rows: {len(result)}")

        if 'event_code' in result.columns:
            events = result[result['event_code'] != 'none']
            print(f"Events detected: {len(events)}")
            if len(events) > 0:
                print(f"\nEvent distribution:")
                print(events['event_label'].value_counts().to_string())

        if 'sell_signal' in result.columns:
            sells = result[result['sell_signal'].fillna(False)]
            if len(sells) > 0:
                print(f"\n>> SELL SIGNALS: {len(sells)}")

        if 'climax_top' in result.columns:
            climax = result[result['climax_top'].fillna(False)]
            if len(climax) > 0:
                print(f"\nWARNING:  CLIMAX TOPS: {len(climax)}")

        # Stage-enriched extras
        if args.mode == 'stage_enriched' and 'w_stage_label' in result.columns:
            stage_adj_count = (result.get('stage_adj', pd.Series(0)) != 0).sum()
            print(f"\n  Stage adjustments: {stage_adj_count} rows modified")
            if 'structure' in result.columns:
                print(f"\n  Options overlay per ticker:")
                overlay_cols = ['symbol', 'w_stage_label', 'hv_20', 'atm_iv',
                                'iv_percentile', 'iv_regime', 'iv_source',
                                'structure', 'dte_range']
                overlay_display = result.drop_duplicates(subset=['symbol'])[
                    [c for c in overlay_cols if c in result.columns]
                ]
                print(overlay_display.to_string(index=False))

        # Show top results
        display_cols = [c for c in [
            args.col_symbol, args.col_date, args.col_close, 'gap_pct',
            'event_label', 'event_score', 'w_stage_label', 'stage_adj',
            'signal', 'sell_signal'
        ] if c in result.columns]

        print(f"\nTop {args.top} results:")
        print(result[display_cols].head(args.top).to_string(index=False))

        print(f"\nSaved to: {output_file}")

    # ── Dashboard + Plots ──
    dash_prefix = os.path.splitext(output_file)[0]

    if args.dashboard:
        if not args.quiet:
            print(f"\nGenerating dashboard (top {args.dashboard_top})...")
        write_dashboard(result, output_prefix=dash_prefix, top_n=args.dashboard_top)

    if args.plots:
        if not args.quiet:
            print("\nGenerating plots...")
        write_plots(result, output_prefix=dash_prefix)

    if not args.quiet and (args.dashboard or args.plots):
        print("\nAll outputs complete.")


if __name__ == '__main__':
    main_cli()
