import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_EVENT_KEYWORDS = [
    'earnings',
    'revenue',
    'profit',
    'guidance',
    'forecast',
    'outlook',
    'margin',
    'eps',
    'analyst',
    'upgrade',
    'downgrade',
    'price target',
    'beats',
    'misses',
    'demand',
    'supply',
    'customer',
    'order',
    'contract',
    'partnership',
    'launch',
    'product',
    'regulatory',
    'lawsuit',
    'antitrust',
    'export',
    'china',
    'covid',
    'coronavirus',
    'pandemic',
]

BOILERPLATE_PATTERNS = [
    r'the views and opinions expressed herein',
    r'do not necessarily reflect those of nasdaq',
    r'copyright',
    r'unauthorized reproduction',
    r'this article originally appeared',
]

BROAD_TITLE_PATTERNS = [
    r'us stocks',
    r'wall st',
    r'wall street',
    r'stock market',
    r'market update',
    r'market wrap',
    r'nasdaq',
    r'dow jones',
    r's&p',
    r'etf',
    r'large inflows?',
    r'large outflows?',
    r'notable etf',
    r'zacks analyst blog',
    r'analyst blog highlights',
    r'\b\d+ stocks? to',
    r'stocks? (?:rise|fall|mixed|move)',
    r'option activity',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build reusable text views for news embedding experiments: title, summary, '
            'selected body sentences, full body, and a cascade text column.'
        )
    )
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--ticker', default='')
    parser.add_argument('--company_name', default='')
    parser.add_argument('--aliases', nargs='*', default=None)
    parser.add_argument('--business_keywords', nargs='*', default=None)
    parser.add_argument('--title_col', default='Article_title')
    parser.add_argument('--body_col', default='Article')
    parser.add_argument(
        '--summary_cols',
        nargs='*',
        default=['Lsa_summary', 'Lexrank_summary', 'Textrank_summary', 'Luhn_summary'],
        help='Summary columns checked in order. The first non-empty summary is used.',
    )
    parser.add_argument(
        '--source_mode',
        choices=['title', 'summary', 'selected_body_sentences', 'full_body', 'cascade'],
        default='cascade',
        help='Which view to copy into output_text_col.',
    )
    parser.add_argument('--output_text_col', default='news_text')
    parser.add_argument('--output_source_col', default='news_text_source')
    parser.add_argument('--max_chars', type=int, default=1200)
    parser.add_argument('--summary_max_chars', type=int, default=900)
    parser.add_argument('--body_sentence_count', type=int, default=3)
    parser.add_argument('--sentence_max_chars', type=int, default=320)
    parser.add_argument(
        '--title_min_score',
        type=float,
        default=2.0,
        help='Cascade uses title only when the title relevance score reaches this value.',
    )
    return parser.parse_args()


def clean_text(value):
    if pd.isna(value):
        return ''
    text = str(value)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate(text, max_chars):
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0].strip()


def compile_terms(terms):
    clean_terms = [term for term in terms if term]
    if not clean_terms:
        return None
    patterns = []
    for term in clean_terms:
        escaped = re.escape(term)
        if re.fullmatch(r'[A-Za-z0-9]{1,5}', term):
            patterns.append(rf'\b{escaped}\b')
        else:
            patterns.append(escaped)
    return re.compile('|'.join(f'(?:{pattern})' for pattern in patterns), re.IGNORECASE)


def compile_patterns(patterns):
    if not patterns:
        return None
    return re.compile('|'.join(f'(?:{pattern})' for pattern in patterns), re.IGNORECASE)


def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    pieces = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for piece in pieces:
        piece = clean_text(piece)
        if len(piece) >= 25:
            out.append(piece)
    return out


def first_non_empty_summary(row, summary_cols, max_chars):
    for col in summary_cols:
        if col not in row.index:
            continue
        text = truncate(row[col], max_chars)
        if text:
            return text, col
    return '', ''


def score_text(text, target_re, business_re, event_re, broad_re, boilerplate_re):
    text = clean_text(text)
    if not text:
        return 0.0
    score = 0.0
    if target_re and target_re.search(text):
        score += 3.0
    if event_re and event_re.search(text):
        score += 2.0
    if business_re and business_re.search(text):
        score += 1.5
    if broad_re and broad_re.search(text):
        score -= 2.0
    if boilerplate_re and boilerplate_re.search(text):
        score -= 3.0
    return score


def selected_body_sentences(row, args, target_re, business_re, event_re, boilerplate_re):
    if args.body_col not in row.index:
        return ''
    sentences = split_sentences(row[args.body_col])
    if not sentences:
        return ''

    scored = []
    for pos, sentence in enumerate(sentences):
        score = score_text(
            sentence,
            target_re=target_re,
            business_re=business_re,
            event_re=event_re,
            broad_re=None,
            boilerplate_re=boilerplate_re,
        )
        if score > 0:
            scored.append((score, pos, truncate(sentence, args.sentence_max_chars)))

    if not scored:
        return ''

    scored = sorted(scored, key=lambda item: (-item[0], item[1]))[:max(args.body_sentence_count, 1)]
    scored = sorted(scored, key=lambda item: item[1])
    return truncate(' '.join(sentence for _, _, sentence in scored), args.max_chars)


def build_alias_terms(args):
    terms = []
    if args.ticker:
        terms.append(args.ticker)
    if args.company_name:
        terms.append(args.company_name)
    if args.aliases:
        terms.extend(args.aliases)
    return terms


def choose_text(row, args, title_score):
    views = {
        'title': row['news_text_title'],
        'summary': row['news_text_summary'],
        'selected_body_sentences': row['news_text_body_sentences'],
        'full_body': row['news_text_full_body'],
    }
    if args.source_mode != 'cascade':
        return views[args.source_mode], args.source_mode

    if views['title'] and title_score >= args.title_min_score:
        return views['title'], 'title'
    if views['summary']:
        return views['summary'], row['news_text_summary_source'] or 'summary'
    if views['selected_body_sentences']:
        return views['selected_body_sentences'], 'selected_body_sentences'
    if views['full_body']:
        return views['full_body'], 'full_body'
    return views['title'], 'title_empty_fallback'


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    target_re = compile_terms(build_alias_terms(args))
    business_re = compile_terms(args.business_keywords or [])
    event_re = compile_terms(DEFAULT_EVENT_KEYWORDS)
    broad_re = compile_patterns(BROAD_TITLE_PATTERNS)
    boilerplate_re = compile_patterns(BOILERPLATE_PATTERNS)

    out = df.copy()
    if args.title_col in out.columns:
        out['news_text_title'] = out[args.title_col].map(lambda value: truncate(value, args.max_chars))
    else:
        out['news_text_title'] = ''

    summaries = out.apply(
        lambda row: first_non_empty_summary(row, args.summary_cols, args.summary_max_chars),
        axis=1,
        result_type='expand',
    )
    out['news_text_summary'] = summaries[0]
    out['news_text_summary_source'] = summaries[1]

    out['news_text_body_sentences'] = out.apply(
        lambda row: selected_body_sentences(row, args, target_re, business_re, event_re, boilerplate_re),
        axis=1,
    )
    if args.body_col in out.columns:
        out['news_text_full_body'] = out[args.body_col].map(lambda value: truncate(value, args.max_chars))
    else:
        out['news_text_full_body'] = ''

    out['news_text_title_score'] = out['news_text_title'].map(
        lambda text: score_text(text, target_re, business_re, event_re, broad_re, boilerplate_re)
    )

    chosen = out.apply(lambda row: choose_text(row, args, row['news_text_title_score']), axis=1, result_type='expand')
    out[args.output_text_col] = chosen[0]
    out[args.output_source_col] = chosen[1]

    out.to_csv(output_csv, index=False)

    print(f'Loaded rows: {len(out)}')
    print(f'Saved: {output_csv}')
    print('chosen_source_counts=')
    print(out[args.output_source_col].value_counts(dropna=False).to_string())
    for col in ['news_text_title', 'news_text_summary', 'news_text_body_sentences', 'news_text_full_body', args.output_text_col]:
        non_empty = out[col].astype(str).str.len().gt(0).sum()
        print(f'{col}_non_empty={non_empty} ({non_empty / max(len(out), 1) * 100:.2f}%)')


if __name__ == '__main__':
    main()
