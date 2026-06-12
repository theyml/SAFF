import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ZS_RELEVANCE_LABELS = [
    'target company specific news',
    'target product or business specific news',
    'sector, supply chain, or competitor context',
    'generic market update, ETF flow, or stock list',
    'unrelated or wrong company news',
]

OUTPUT_COLS = [
    'zs_rel_company_specific_prob',
    'zs_rel_product_business_prob',
    'zs_rel_sector_competitor_prob',
    'zs_rel_market_noise_prob',
    'zs_rel_unrelated_prob',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append target-conditioned zero-shot relevance columns to a news embedding CSV.'
    )
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--keep_output_csv', default=None, help='Optional hard-filtered CSV containing only kept rows.')
    parser.add_argument('--text_col', default='Article_title')
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--company_name', required=True)
    parser.add_argument('--sector', default='')
    parser.add_argument('--business_description', default='')
    parser.add_argument(
        '--model_name_or_path',
        default='MoritzLaurer/ModernBERT-large-zeroshot-v2.0',
        help='HF model name or local path for zero-shot classification.',
    )
    parser.add_argument(
        '--hypothesis_template',
        default='This text is {}.',
        help='Zero-shot hypothesis template passed to the classifier.',
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', default=None, help='cuda / cpu. Default: auto-detect.')
    parser.add_argument('--direct_threshold', type=float, default=0.45)
    parser.add_argument('--sector_threshold', type=float, default=0.60)
    parser.add_argument('--noise_threshold', type=float, default=0.30)
    parser.add_argument('--sector_weight', type=float, default=0.35)
    parser.add_argument('--market_noise_weight', type=float, default=0.05)
    return parser.parse_args()


def is_local_path(model_name_or_path):
    return os.path.exists(model_name_or_path) or str(model_name_or_path).startswith(('.', '/', '~'))


def resolve_device(device):
    if device:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    row_sum = values.sum(axis=1, keepdims=True)
    return values / np.clip(row_sum, 1e-8, None)


def build_premises(df, args):
    titles = df[args.text_col].fillna('').astype(str).tolist()
    prefix = (
        f'Target company: {args.company_name}\n'
        f'Ticker: {args.ticker}\n'
        f'Sector: {args.sector}\n'
        f'Business description: {args.business_description}\n\n'
        'News title:\n'
    )
    suffix = '\n\nTask: Classify how specifically this news is relevant to the target company stock price.'
    return [prefix + title + suffix for title in titles]


def build_zero_shot_probs(premises, args, device):
    local_only = is_local_path(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=local_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    label2id = {str(k).lower(): int(v) for k, v in getattr(model.config, 'label2id', {}).items()}
    entailment_idx = label2id.get('entailment')
    if entailment_idx is None:
        entailment_idx = label2id.get('entailment_label')
    if entailment_idx is None:
        # Most NLI checkpoints use [contradiction, neutral, entailment].
        entailment_idx = int(getattr(model.config, 'num_labels', 3)) - 1

    hypotheses = [args.hypothesis_template.format(label) for label in ZS_RELEVANCE_LABELS]

    probs = []
    for start in tqdm(range(0, len(premises), args.batch_size), desc='Zero-shot relevance', unit='batch'):
        batch = premises[start:start + args.batch_size]
        paired_premises = []
        paired_hypotheses = []
        for premise in batch:
            for hypothesis in hypotheses:
                paired_premises.append(premise)
                paired_hypotheses.append(hypothesis)

        encoded = tokenizer(
            paired_premises,
            paired_hypotheses,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            label_probs = F.softmax(logits, dim=-1)
            entailment_scores = label_probs[:, entailment_idx]
            batch_probs = entailment_scores.reshape(len(batch), len(ZS_RELEVANCE_LABELS))
            probs.append(batch_probs.detach().cpu().numpy())

    return normalize_rows(np.concatenate(probs, axis=0))


def append_relevance_columns(df, probs, args):
    out = df.copy()
    for col, values in zip(OUTPUT_COLS, probs.T):
        out[col] = values

    direct_prob = out['zs_rel_company_specific_prob'] + out['zs_rel_product_business_prob']
    noise_prob = out['zs_rel_market_noise_prob'] + out['zs_rel_unrelated_prob']
    out['zs_rel_direct_prob'] = direct_prob
    out['zs_rel_noise_prob'] = noise_prob

    label_idx = probs.argmax(axis=1)
    out['zs_rel_label'] = [ZS_RELEVANCE_LABELS[int(i)] for i in label_idx]
    out['zs_rel_confidence'] = probs.max(axis=1)
    out['zs_rel_keep_hard'] = (
        (direct_prob >= args.direct_threshold)
        | (
            (out['zs_rel_sector_competitor_prob'] >= args.sector_threshold)
            & (noise_prob < args.noise_threshold)
        )
    ).astype(int)
    out['zs_rel_weight'] = (
        out['zs_rel_company_specific_prob']
        + out['zs_rel_product_business_prob']
        + float(args.sector_weight) * out['zs_rel_sector_competitor_prob']
        + float(args.market_noise_weight) * out['zs_rel_market_noise_prob']
    )
    out['zs_rel_source'] = 'zero_shot_relevance'
    return out


def print_summary(out):
    total = len(out)
    kept = int(out['zs_rel_keep_hard'].sum())
    print(f'total_rows={total}')
    print(f'kept_rows={kept} ({kept / max(total, 1) * 100:.2f}%)')
    print('label_counts=')
    print(out['zs_rel_label'].value_counts(dropna=False).to_string())
    print('mean_probs=')
    for col in OUTPUT_COLS + ['zs_rel_direct_prob', 'zs_rel_noise_prob', 'zs_rel_weight']:
        print(f'{col}={out[col].mean():.4f}')


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_output_csv:
        Path(args.keep_output_csv).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f'text_col={args.text_col!r} not found in {input_csv}')

    device = resolve_device(args.device)
    premises = build_premises(df, args)
    probs = build_zero_shot_probs(premises, args, device)
    out = append_relevance_columns(df, probs, args)
    print_summary(out)
    out.to_csv(output_csv, index=False)
    if args.keep_output_csv:
        out.loc[out['zs_rel_keep_hard'].astype(bool)].to_csv(args.keep_output_csv, index=False)


if __name__ == '__main__':
    main()
