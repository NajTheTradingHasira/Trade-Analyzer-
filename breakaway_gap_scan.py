import pandas as pd
import numpy as np

INPUT_FILE = 'ohlcv.csv'
OUTPUT_FILE = 'breakaway_gap_watchlist.csv'


def _atr(g, high='high', low='low', close='close', n=14):
    tr1 = g[high] - g[low]
    tr2 = (g[high] - g[close].shift(1)).abs()
    tr3 = (g[low] - g[close].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def breakaway_gap_scan(df, symbol_col='symbol', date_col='date', open_col='open', high_col='high', low_col='low', close_col='close', volume_col='volume'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])
    out = []

    for sym, g in df.groupby(symbol_col):
        g = g.copy().reset_index(drop=True)
        g['prev_close'] = g[close_col].shift(1)
        g['gap_pct'] = (g[open_col] / g['prev_close']) - 1
        g['gap_up'] = g['gap_pct'] > 0.02
        g['gap_down'] = g['gap_pct'] < -0.02
        g['vol_ma_50'] = g[volume_col].rolling(50).mean()
        g['vol_spike'] = g[volume_col] >= 1.5 * g['vol_ma_50']
        g['atr14'] = _atr(g, high_col, low_col, close_col, 14)
        g['atr_pct'] = g['atr14'] / g[close_col]
        g['range_pct'] = (g[high_col] - g[low_col]) / g['prev_close']

        base_low_20 = g[low_col].rolling(20).min().shift(1)
        base_high_20 = g[high_col].rolling(20).max().shift(1)
        base_low_60 = g[low_col].rolling(60).min().shift(1)
        base_high_60 = g[high_col].rolling(60).max().shift(1)

        g['breakout_20d_high'] = g[open_col] > base_high_20
        g['breakdown_20d_low'] = g[open_col] < base_low_20
        g['breakout_60d_high'] = g[open_col] > base_high_60
        g['breakdown_60d_low'] = g[open_col] < base_low_60

        g['close_holds_gap_up'] = g[low_col] > g['prev_close']
        g['close_holds_gap_down'] = g[high_col] < g['prev_close']
        g['fills_gap_intraday'] = (g[low_col] <= g['prev_close']) & (g[high_col] >= g['prev_close'])

        g['trend_20'] = g[close_col] > g[close_col].rolling(20).mean()
        g['trend_50'] = g[close_col] > g[close_col].rolling(50).mean()
        g['trend_200'] = g[close_col] > g[close_col].rolling(200).mean()
        g['new_20d_high'] = g[close_col] >= g[close_col].rolling(20).max()
        g['new_60d_high'] = g[close_col] >= g[close_col].rolling(60).max()
        g['new_20d_low'] = g[close_col] <= g[close_col].rolling(20).min()
        g['new_60d_low'] = g[close_col] <= g[close_col].rolling(60).min()

        g['bullish_breakaway'] = (
            g['gap_up'] &
            g['breakout_20d_high'] &
            g['vol_spike'] &
            g['trend_20'].fillna(False)
        )
        g['bearish_breakaway'] = (
            g['gap_down'] &
            g['breakdown_20d_low'] &
            g['vol_spike'] &
            (~g['trend_20'].fillna(False))
        )

        g['bull_score'] = (
            2.0 * g['bullish_breakaway'].fillna(False).astype(int) +
            1.0 * g['gap_up'].fillna(False).astype(int) +
            1.0 * g['vol_spike'].fillna(False).astype(int) +
            1.0 * g['breakout_60d_high'].fillna(False).astype(int) +
            0.5 * g['trend_50'].fillna(False).astype(int) +
            0.5 * g['trend_200'].fillna(False).astype(int) +
            0.5 * g['new_20d_high'].fillna(False).astype(int)
        )

        g['bear_score'] = (
            2.0 * g['bearish_breakaway'].fillna(False).astype(int) +
            1.0 * g['gap_down'].fillna(False).astype(int) +
            1.0 * g['vol_spike'].fillna(False).astype(int) +
            1.0 * g['breakdown_60d_low'].fillna(False).astype(int) +
            0.5 * (~g['trend_50'].fillna(False)).astype(int) +
            0.5 * (~g['trend_200'].fillna(False)).astype(int) +
            0.5 * g['new_20d_low'].fillna(False).astype(int)
        )

        g['direction'] = np.where(g['bull_score'] >= g['bear_score'], 'bullish', 'bearish')
        g['score'] = np.maximum(g['bull_score'], g['bear_score'])
        g['signal'] = np.where(g['bullish_breakaway'], 'bullish_breakaway', np.where(g['bearish_breakaway'], 'bearish_breakaway', 'none'))
        g['symbol'] = sym
        out.append(g)

    res = pd.concat(out, ignore_index=True)
    cols = [symbol_col, date_col, open_col, high_col, low_col, close_col, volume_col, 'gap_pct', 'signal', 'direction', 'score', 'bullish_breakaway', 'bearish_breakaway', 'gap_up', 'gap_down', 'vol_spike', 'breakout_20d_high', 'breakdown_20d_low', 'breakout_60d_high', 'breakdown_60d_low', 'close_holds_gap_up', 'close_holds_gap_down']
    res = res[cols].sort_values(['score', date_col], ascending=[False, False])
    return res


if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE)
    result = breakaway_gap_scan(df)
    result.to_csv(OUTPUT_FILE, index=False)
    print(result.head(30).to_string(index=False))
