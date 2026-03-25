"""
Patch 3: Composite Event Labeling
Adds: event_label + event_score

Combines gap type, fill risk, earnings proximity, and volume/trend
into a single composite event classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from breakaway_gap_scan import breakaway_gap_scan
from patches.add_gap_fill_risk import add_gap_fill_risk
from patches.add_post_earnings_flag import post_earnings_flag_scan


EVENT_LABELS = {
    'earnings_breakaway_bull': 'Earnings Breakaway Gap (Bullish)',
    'earnings_breakaway_bear': 'Earnings Breakaway Gap (Bearish)',
    'breakaway_bull': 'Breakaway Gap (Bullish)',
    'breakaway_bear': 'Breakaway Gap (Bearish)',
    'continuation_bull': 'Continuation Gap (Bullish)',
    'continuation_bear': 'Continuation Gap (Bearish)',
    'exhaustion_bull': 'Exhaustion Gap (Bullish) — High Fill Risk',
    'exhaustion_bear': 'Exhaustion Gap (Bearish) — High Fill Risk',
    'common_gap': 'Common Gap — No Conviction',
    'none': 'No Event',
}


def add_composite_event(df, symbol_col='symbol', date_col='date'):
    """
    Classify each row into a composite event type and score.

    Adds:
        - event_label: str — human-readable event classification
        - event_code: str — machine-readable event key
        - event_score: float — composite conviction score (0-10)
    """
    df = df.copy()

    df['event_code'] = 'none'
    df['event_label'] = EVENT_LABELS['none']
    df['event_score'] = 0.0

    # Reset index so .loc works reliably
    df = df.reset_index(drop=True)

    has_gap = df.get('gap_up', pd.Series(False, index=df.index)).fillna(False) | df.get('gap_down', pd.Series(False, index=df.index)).fillna(False)
    is_earnings = df.get('is_earnings_gap', pd.Series(False, index=df.index)).fillna(False)
    is_bull_breakaway = df.get('bullish_breakaway', pd.Series(False, index=df.index)).fillna(False)
    is_bear_breakaway = df.get('bearish_breakaway', pd.Series(False, index=df.index)).fillna(False)
    fill_risk = df.get('gap_fill_risk', pd.Series('n/a', index=df.index))
    vol_spike = df.get('vol_spike', pd.Series(False, index=df.index)).fillna(False)
    trend_50 = df.get('trend_50', pd.Series(False, index=df.index)).fillna(False)
    trend_200 = df.get('trend_200', pd.Series(False, index=df.index)).fillna(False)
    gap_up = df.get('gap_up', pd.Series(False, index=df.index)).fillna(False)
    gap_down = df.get('gap_down', pd.Series(False, index=df.index)).fillna(False)
    gap_pct = df.get('gap_pct', pd.Series(0.0, index=df.index)).abs()
    score_s = df.get('score', pd.Series(0.0, index=df.index))

    for idx in df.index:
        if not has_gap.loc[idx]:
            continue

        base_score = float(score_s.loc[idx])

        # ── Earnings Breakaway (highest priority) ──
        if is_earnings.loc[idx]:
            if is_bull_breakaway.loc[idx]:
                df.loc[idx, 'event_code'] = 'earnings_breakaway_bull'
                base_score += 3.0
            elif is_bear_breakaway.loc[idx]:
                df.loc[idx, 'event_code'] = 'earnings_breakaway_bear'
                base_score += 3.0
            elif gap_up.loc[idx]:
                df.loc[idx, 'event_code'] = 'continuation_bull'
                base_score += 1.5
            else:
                df.loc[idx, 'event_code'] = 'continuation_bear'
                base_score += 1.5

        # ── Non-earnings Breakaway ──
        elif is_bull_breakaway.loc[idx]:
            df.loc[idx, 'event_code'] = 'breakaway_bull'
            base_score += 2.0

        elif is_bear_breakaway.loc[idx]:
            df.loc[idx, 'event_code'] = 'breakaway_bear'
            base_score += 2.0

        # ── High fill risk = possible exhaustion gap ──
        elif fill_risk.loc[idx] == 'high' and gap_up.loc[idx]:
            df.loc[idx, 'event_code'] = 'exhaustion_bull'
            base_score += 0.5

        elif fill_risk.loc[idx] == 'high' and gap_down.loc[idx]:
            df.loc[idx, 'event_code'] = 'exhaustion_bear'
            base_score += 0.5

        # ── Continuation gap ──
        elif vol_spike.loc[idx] and gap_up.loc[idx] and trend_50.loc[idx]:
            df.loc[idx, 'event_code'] = 'continuation_bull'
            base_score += 1.0

        elif vol_spike.loc[idx] and gap_down.loc[idx] and not trend_50.loc[idx]:
            df.loc[idx, 'event_code'] = 'continuation_bear'
            base_score += 1.0

        # ── Common gap ──
        else:
            df.loc[idx, 'event_code'] = 'common_gap'

        # ── Score bonuses ──
        if vol_spike.loc[idx]:
            base_score += 0.5
        if trend_200.loc[idx] and gap_up.loc[idx]:
            base_score += 0.5
        if gap_pct.loc[idx] > 0.05:
            base_score += 1.0
        if fill_risk.loc[idx] == 'low':
            base_score += 0.5

        df.loc[idx, 'event_score'] = round(min(base_score, 10.0), 1)

    # Apply labels
    df['event_label'] = df['event_code'].map(EVENT_LABELS).fillna('Unknown')

    return df


if __name__ == '__main__':
    print("Patch 3: Adding composite event labels + scores...")
    df = pd.read_csv('ohlcv.csv')
    result = breakaway_gap_scan(df)
    result = add_gap_fill_risk(result)
    result = post_earnings_flag_scan(result)
    result = add_composite_event(result)

    events = result[result['event_code'] != 'none']
    print(f"\nEvents classified: {len(events)}")
    print(f"\nEvent distribution:")
    print(events['event_label'].value_counts().to_string())
    print(f"\nTop events by score:")
    print(events[['symbol', 'date', 'event_label', 'event_score', 'gap_pct']].nlargest(15, 'event_score').to_string(index=False))

    result.to_csv('breakaway_gap_watchlist.csv', index=False)
    print("\nSaved to breakaway_gap_watchlist.csv")
