# APEX Trade Analyzer

Multi-dimensional event scanner and signal ranker. Detects breakaway gaps, climax tops, Weinstein stage transitions, options overlays, and composite scoring across a watchlist of tickers.

---

## Quick Start

```bash
# Install from source
pip install -e .

# Or use requirements.txt directly
pip install -r requirements.txt

# Run the default unified scan
trade-analyzer --input ohlcv.csv --mode unified

# Or invoke directly
python patches/add_cli_runner.py --input ohlcv.csv --mode unified
```

---

## CLI Modes

| Mode                | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| `unified`           | Full event scan — breakaway gaps, climax tops, composite scoring  |
| `breakaway`         | Breakaway gap detection with gap-fill risk and composite events   |
| `climax`            | Climax top scan — exhaustion signals and sell alerts               |
| `gaps-only`         | Raw breakaway gap scan (no composite overlay)                     |
| `stage_enriched`    | Weinstein stage classification + options overlay + stage-adjusted scores |
| `backtest`          | Historical backtest of event signals with forward-return tables   |
| `absolute_strength` | Relative & absolute strength monitor with tier alerts             |
| `theme_momentum`    | Sector/theme momentum time-series and rotation charts             |
| `master`            | Master score combining all sub-scores (events, stage, strength)   |
| `master_matrix`     | Full cross-dimensional matrix with auto-generated dashboard       |
| `master_dashboard`  | Standalone dashboard generator from an existing master_matrix CSV |
| `linear_tracker`    | Linear regression channel tracker with trend quality metrics      |
| `linear_regime`     | Regime detection via linear regression slope and R-squared        |
| `ticker_dashboard`  | Single-ticker deep-dive dashboard (requires `--symbol`)           |

---

## Common Flags

| Flag              | Default    | Description                                      |
|-------------------|------------|--------------------------------------------------|
| `--input, -i`     | (required) | Input CSV file with OHLCV data                   |
| `--output, -o`    | auto-named | Output CSV file path                              |
| `--mode, -m`      | `unified`  | Scanner mode (see table above)                    |
| `--symbol, -s`    | all        | Filter to a single ticker symbol                  |
| `--min-score`     | `0.0`      | Minimum event score threshold                     |
| `--top, -n`       | `50`       | Number of top results to display                  |
| `--events-only`   | off        | Only output rows with detected events             |
| `--dashboard`     | off        | Export styled HTML dashboard + CSV                |
| `--plots`         | off        | Generate Plotly gap histogram + scatter plot       |
| `--quiet, -q`     | off        | Suppress progress output                          |
| `--date-from`     |            | Start date filter (YYYY-MM-DD)                    |
| `--date-to`       |            | End date filter (YYYY-MM-DD)                      |
| `--dashboard-top` | `20`       | Number of events in dashboard output              |

Column name overrides: `--col-symbol`, `--col-date`, `--col-open`, `--col-high`, `--col-low`, `--col-close`, `--col-volume`.

---

## Example Commands

```bash
# Unified scan with dashboard and plots
trade-analyzer -i ohlcv.csv -m unified --dashboard --plots

# Breakaway gaps, minimum score 4, top 20
trade-analyzer -i ohlcv.csv -m breakaway --min-score 4 --top 20

# Climax top scan for a single symbol
trade-analyzer -i ohlcv.csv -m climax --symbol NVDA

# Stage-enriched with Weinstein classification
trade-analyzer -i ohlcv.csv -m stage_enriched --events-only --dashboard

# Backtest event signals
trade-analyzer -i ohlcv.csv -m backtest

# Absolute strength monitor
trade-analyzer -i ohlcv.csv -m absolute_strength

# Theme/sector momentum
trade-analyzer -i ohlcv.csv -m theme_momentum

# Master matrix (full pipeline)
trade-analyzer -i ohlcv.csv -m master_matrix --top 50

# Master dashboard from existing CSV
trade-analyzer -i master_matrix_results.csv -m master_dashboard

# Linear regression tracker
trade-analyzer -i ohlcv.csv -m linear_tracker

# Linear regime detection
trade-analyzer -i ohlcv.csv -m linear_regime

# Single-ticker deep-dive
trade-analyzer -i ohlcv.csv -m ticker_dashboard --symbol AAPL
```

---

## Alert Setup

### Slack

1. Create a Slack Incoming Webhook at https://api.slack.com/messaging/webhooks.
2. Set `SLACK_WEBHOOK_URL` in your `.env` file.
3. Run alerts:

```bash
python patches/alerts.py --slack-only --min-score 7
```

### Email

1. Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, and `ALERT_EMAIL_TO` in `.env`.
2. Run alerts:

```bash
python patches/alerts.py --email-only --min-score 5
```

### Scheduled Alerts (Windows Task Scheduler)

Use the included PowerShell scripts to register a daily morning scan:

```powershell
.\setup_morning_alert.ps1
.\setup_scheduled_alert.ps1
```

### Dry Run

Preview alerts without sending:

```bash
python patches/alerts.py --dry-run --min-score 7
```

---

## Docker Usage

```bash
# Build the image
docker build -t trade-analyzer .

# Run the daily alert pipeline
docker run --env-file .env trade-analyzer

# Override command for an interactive scan
docker run --env-file .env -v $(pwd):/app trade-analyzer \
    python patches/add_cli_runner.py -i ohlcv.csv -m master_matrix
```

---

## Architecture

The scanner is built as a layered patch system. Each patch in `patches/` adds a capability on top of the base scanner:

| #  | File                         | Capability                                       |
|----|------------------------------|--------------------------------------------------|
| 0  | `breakaway_gap_scan.py`      | Base breakaway gap detection                     |
| 1  | `add_gap_fill_risk.py`       | Gap-fill probability scoring                     |
| 2  | `add_post_earnings_flag.py`  | Post-earnings gap flagging                       |
| 3  | `add_composite_event.py`     | Composite event score aggregation                |
| 4  | `add_unified_scanner.py`     | Unified scan + climax top detection              |
| 5  | `add_cli_runner.py`          | CLI runner with argparse                         |
| 6  | `add_dashboard.py`           | HTML dashboard + Plotly plot generation           |
| 7  | `add_stage_enriched.py`      | Weinstein stage classification + options overlay  |
| 8  | `options_overlay.py`         | IV regime, structure recommendation, DTE ranges  |
| 9  | `iv_provider.py`             | Implied volatility data provider                 |
| 10 | `add_backtester.py`          | Historical backtest engine                       |
| 11 | `add_comparative_strength.py`| Comparative / relative strength scoring          |
| 12 | `add_absolute_strength.py`   | Absolute strength composite scoring              |
| 13 | `add_as_monitor.py`          | Absolute strength monitor mode                   |
| 14 | `add_as_alerts.py`           | Strength-based alert generation                  |
| 15 | `add_theme_momentum.py`      | Sector/theme momentum time-series                |
| 16 | `add_master_score.py`        | Master score aggregation                         |
| 17 | `add_master_matrix.py`       | Cross-dimensional scoring matrix                 |
| 18 | `add_master_dashboard.py`    | Master dashboard HTML generator                  |
| 19 | `add_linear_tracker.py`      | Linear regression channel tracker                |
| 20 | `add_linear_regime.py`       | Regime detection via regression metrics           |
| 21 | `add_rank_change_alerts.py`  | Rank-change alert detection                      |
| 22 | `add_ticker_dashboard.py`    | Single-ticker deep-dive dashboard                |
| -- | `alerts.py`                  | Standalone alert engine (Slack + email)           |

---

## Environment Variables

| Variable             | Required | Description                                    |
|----------------------|----------|------------------------------------------------|
| `SLACK_WEBHOOK_URL`  | No       | Slack incoming webhook URL for alert delivery  |
| `SMTP_HOST`          | No       | SMTP server hostname (e.g., smtp.gmail.com)    |
| `SMTP_PORT`          | No       | SMTP server port (e.g., 587)                   |
| `SMTP_USER`          | No       | SMTP username / sender email                   |
| `SMTP_PASS`          | No       | SMTP password or app password                  |
| `ALERT_EMAIL_TO`     | No       | Recipient email for alert delivery             |
| `ANTHROPIC_API_KEY`  | No       | Anthropic API key for AI-assisted analysis     |
| `PERPLEXITY_API_KEY` | No       | Perplexity API key for research queries        |

---

## Node.js Trade Analyzer Suite

The Node.js frontend (server.js, public/) provides 10 live analyst modules powered by Claude and Perplexity. See the original setup instructions:

```bash
npm install
cp .env.example .env
# Edit .env with your API keys
npm start
# Open http://localhost:3000
```

---

## License

MIT
