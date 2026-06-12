import argparse
import re

import pandas as pd


DEFAULT_TARGET_ALIASES = {
    'AMD': [
        r'\bamd\b',
        'advanced micro devices',
        r'\bryzen\b',
        r'\bradeon\b',
        r'\bepyc\b',
        r'\binstinct\b',
        'threadripper',
        'xilinx',
    ],
}

DEFAULT_COMPETITOR_PATTERNS = [
    r'\bnvidia\b',
    r'\bnvda\b',
    r'\bintel\b',
    r'\bintc\b',
    r'\bqualcomm\b',
    r'\bqcom\b',
    r'\bmicron\b',
    r'\bmu\b',
    r'\btsmc\b',
    r'\btsm\b',
    r'\bbroadcom\b',
    r'\bavgo\b',
    r'\bmarvell\b',
    r'\bmrv[ l]?\b',
    r'\bamat\b',
    r'\basml\b',
]

BROAD_MARKET_PATTERNS = [
    'us stocks',
    'wall st',
    'wall street',
    'stock market',
    'market update',
    'mid-day market',
    'pre-market',
    'premarket',
    'after-hours',
    'after hours',
    'nasdaq',
    'dow jones',
    r's&p',
    r's\&p',
    'russell',
    'futures',
    r'stocks (?:rise|fall|mixed|move)',
    'tech stocks',
    'semiconductor stocks',
    'chip stocks',
    'analyst blog highlights',
    'zacks analyst blog',
    'notable etf',
    'large inflows',
    'large outflows',
    'etf',
    'soxl',
    'soxx',
    'smh',
    'qqq',
    'arkk',
    'sector update',
]

EVENT_PATTERNS = [
    'earnings',
    'revenue',
    'profit',
    'guidance',
    'forecast',
    'outlook',
    'margin',
    'data center',
    'ai chip',
    'gpu',
    'cpu',
    'ryzen',
    'epyc',
    'radeon',
    'instinct',
    'launch',
    'unveil',
    'release',
    'order',
    'customer',
    'contract',
    'partnership',
    'acquisition',
    'merger',
    'rating',
    'upgrade',
    'downgrade',
    'price target',
    'analyst',
    'supply',
    'demand',
    'export',
    'china',
    'regulatory',
    'lawsuit',
    'beats',
    'misses',
]

TICKER_STOPWORDS = {
    'AMD',
    'CEO',
    'CFO',
    'SEC',
    'ETF',
    'ETFS',
    'USA',
    'US',
    'AI',
    'CPU',
    'GPU',
    'IPO',
    'EPS',
}


def compile_any(patterns):
    return re.compile('|'.join(f'(?:{p})' for p in patterns), re.IGNORECASE)


def count_other_tickers(title):
    tickers = re.findall(r'\b[A-Z]{2,5}\b', str(title))
    return sum(1 for ticker in tickers if ticker not in TICKER_STOPWORDS)


def build_quality_columns(df, ticker, text_col):
    aliases = DEFAULT_TARGET_ALIASES.get(ticker.upper(), [rf'\b{re.escape(ticker)}\b'])
    target_re = compile_any(aliases)
    competitor_re = compile_any(DEFAULT_COMPETITOR_PATTERNS)
    broad_re = compile_any(BROAD_MARKET_PATTERNS)
    event_re = compile_any(EVENT_PATTERNS)

    titles = df[text_col].fillna('').astype(str)
    direct = titles.str.contains(target_re, regex=True)
    broad = titles.str.contains(broad_re, regex=True)
    competitor = titles.str.contains(competitor_re, regex=True)
    event = titles.str.contains(event_re, regex=True)
    other_ticker_count = titles.map(count_other_tickers)
    multi_ticker = other_ticker_count >= 2

    out = df.copy()
    out['title_quality_direct_match'] = direct.astype(int)
    out['title_quality_broad_market'] = broad.astype(int)
    out['title_quality_competitor_mention'] = competitor.astype(int)
    out['title_quality_event_signal'] = event.astype(int)
    out['title_quality_other_ticker_count'] = other_ticker_count
    out['title_quality_multi_ticker'] = multi_ticker.astype(int)

    score = (
        direct.astype(float) * 3.0
        + event.astype(float) * 2.0
        - broad.astype(float) * 2.0
        - multi_ticker.astype(float)
        - (competitor & ~direct).astype(float) * 2.0
        - (competitor & direct).astype(float) * 0.5
    )
    out['title_quality_score'] = score
    out['title_quality_bucket'] = pd.cut(
        score,
        bins=[-999, 0, 2, 999],
        labels=['low', 'medium', 'high'],
    ).astype(str)
    return out


def mask_for_mode(df, mode):
    direct = df['title_quality_direct_match'].astype(bool)
    broad = df['title_quality_broad_market'].astype(bool)
    event = df['title_quality_event_signal'].astype(bool)
    multi = df['title_quality_multi_ticker'].astype(bool)
    score = df['title_quality_score']

    if mode == 'annotate_all':
        return pd.Series(True, index=df.index)
    if mode == 'direct_not_broad':
        return direct & ~broad
    if mode == 'direct_event_not_broad':
        return direct & event & ~broad
    if mode == 'direct_event_clean':
        return direct & event & ~broad & ~multi
    if mode == 'score_positive':
        return score > 0
    if mode == 'high':
        return df['title_quality_bucket'].eq('high')
    raise ValueError(f'Unknown mode: {mode}')


def print_summary(df, kept):
    total = len(df)
    print(f'total_rows={total}')
    for col in [
        'title_quality_direct_match',
        'title_quality_broad_market',
        'title_quality_multi_ticker',
        'title_quality_competitor_mention',
        'title_quality_event_signal',
    ]:
        count = int(df[col].sum())
        print(f'{col}={count} ({count / max(total, 1) * 100:.2f}%)')
    print('bucket_counts=')
    print(df['title_quality_bucket'].value_counts(dropna=False).to_string())
    print(f'kept_rows={int(kept.sum())} ({kept.mean() * 100:.2f}%)')


def parse_args():
    parser = argparse.ArgumentParser(description='Annotate or filter news rows by title-level target relevance.')
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--text_col', default='Article_title')
    parser.add_argument(
        '--mode',
        default='direct_not_broad',
        choices=[
            'annotate_all',
            'direct_not_broad',
            'direct_event_not_broad',
            'direct_event_clean',
            'score_positive',
            'high',
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f'text_col={args.text_col!r} not found in {args.input_csv}')
    scored = build_quality_columns(df, args.ticker, args.text_col)
    kept = mask_for_mode(scored, args.mode)
    print_summary(scored, kept)
    scored.loc[kept].to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
