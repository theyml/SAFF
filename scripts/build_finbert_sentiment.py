import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append FinBERT headline sentiment probabilities to an existing news CSV.'
    )
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--text_col', default='Article_title')
    parser.add_argument('--model_name_or_path', default='ProsusAI/finbert')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_device(device):
    if device:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def is_local_path(path_or_repo):
    return os.path.exists(path_or_repo) or str(path_or_repo).startswith(('.', '/', '~'))


def label_index(config, name, fallback):
    label2id = {str(k).lower(): int(v) for k, v in getattr(config, 'label2id', {}).items()}
    if name in label2id:
        return label2id[name]
    for label, idx in label2id.items():
        if name in label:
            return idx
    return fallback


def main():
    args = parse_args()
    if os.path.exists(args.output_csv) and not args.overwrite:
        print(f'Output exists, skipping: {args.output_csv}')
        return

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f'text_col={args.text_col!r} not found in {args.input_csv}')

    device = resolve_device(args.device)
    local_only = is_local_path(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        local_files_only=local_only,
    ).to(device)
    model.eval()

    pos_idx = label_index(model.config, 'positive', 0)
    neg_idx = label_index(model.config, 'negative', 1)
    neu_idx = label_index(model.config, 'neutral', 2)
    id2label = {int(k): str(v).lower() for k, v in getattr(model.config, 'id2label', {}).items()}

    texts = df[args.text_col].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().tolist()
    probs = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc='FinBERT sentiment', unit='batch'):
        batch = texts[start:start + args.batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    labels = probs.argmax(axis=1)
    out = df.copy()
    out['finbert_sent_positive_prob'] = probs[:, pos_idx]
    out['finbert_sent_negative_prob'] = probs[:, neg_idx]
    out['finbert_sent_neutral_prob'] = probs[:, neu_idx]
    out['finbert_sentiment_score'] = out['finbert_sent_positive_prob'] - out['finbert_sent_negative_prob']
    out['finbert_sentiment_abs_score'] = out['finbert_sentiment_score'].abs()
    out['finbert_sentiment_confidence'] = probs.max(axis=1)
    out['finbert_sentiment_label'] = [id2label.get(int(i), str(int(i))) for i in labels]
    out['finbert_sentiment_source'] = args.model_name_or_path

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f'Wrote: {args.output_csv}')
    print(f'Rows: {len(out)}')
    print(
        'Mean probs: '
        f"positive={out['finbert_sent_positive_prob'].mean():.4f}, "
        f"negative={out['finbert_sent_negative_prob'].mean():.4f}, "
        f"neutral={out['finbert_sent_neutral_prob'].mean():.4f}"
    )
    print(f"Mean score: {out['finbert_sentiment_score'].mean():+.4f}")
    print('Label counts:')
    print(out['finbert_sentiment_label'].value_counts(dropna=False).to_string())


if __name__ == '__main__':
    main()
