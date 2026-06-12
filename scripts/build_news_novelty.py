import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Build offline novelty scores for precomputed news embeddings.')
    parser.add_argument('--input_csv', type=str, required=True, help='Input news CSV with news_emb_* columns.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV with an added novelty column.')
    parser.add_argument('--time_col', type=str, default='Date', help='Timestamp column in the CSV.')
    parser.add_argument('--group_col', type=str, default='Stock_symbol',
                        help='Optional grouping column such as ticker. Empty string disables grouping.')
    parser.add_argument('--embed_prefix', type=str, default='news_emb_', help='Prefix of embedding columns.')
    parser.add_argument('--lookback_days', type=int, default=30,
                        help='Lookback window in days used to search for similar past news.')
    parser.add_argument('--novelty_col', type=str, default='novelty_score',
                        help='Name of the novelty column to add.')
    parser.add_argument('--min_novelty', type=float, default=0.0,
                        help='Lower clip bound for the final novelty score.')
    parser.add_argument('--max_novelty', type=float, default=1.0,
                        help='Upper clip bound for the final novelty score.')
    return parser.parse_args()


def normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return x / norms


def compute_group_novelty(group_df, embed_cols, time_col, lookback_days):
    group_df = group_df.sort_values(time_col).copy()
    timestamps = pd.to_datetime(group_df[time_col], errors='coerce')
    emb = group_df[embed_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    emb = normalize_rows(emb)

    novelty = np.ones(len(group_df), dtype=np.float32)
    recent_indices = []

    for i, ts in enumerate(timestamps):
        if pd.isna(ts):
            novelty[i] = 1.0
            continue

        cutoff = ts - pd.Timedelta(days=lookback_days)
        while recent_indices and timestamps.iloc[recent_indices[0]] < cutoff:
            recent_indices.pop(0)

        if recent_indices:
            past_emb = emb[recent_indices]
            max_sim = float(np.max(past_emb @ emb[i]))
            novelty[i] = 1.0 - max_sim
        else:
            novelty[i] = 1.0

        recent_indices.append(i)

    group_df['_novelty_tmp'] = novelty
    return group_df


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.time_col not in df.columns:
        raise ValueError(f'time_col={args.time_col!r} not found in {input_csv}')

    embed_cols = [c for c in df.columns if c.startswith(args.embed_prefix)]
    if not embed_cols:
        raise ValueError(f'No embedding columns found with prefix {args.embed_prefix!r} in {input_csv}')

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors='coerce')
    df = df.dropna(subset=[args.time_col]).copy()

    group_col = args.group_col.strip()
    if group_col and group_col in df.columns:
        processed = []
        for _, group_df in df.groupby(group_col, sort=False):
            processed.append(compute_group_novelty(group_df, embed_cols, args.time_col, args.lookback_days))
        out_df = pd.concat(processed, axis=0).sort_index()
    else:
        out_df = compute_group_novelty(df, embed_cols, args.time_col, args.lookback_days)

    novelty = out_df['_novelty_tmp'].to_numpy(dtype=np.float32)
    novelty = np.clip(novelty, args.min_novelty, args.max_novelty)
    out_df[args.novelty_col] = novelty
    out_df = out_df.drop(columns=['_novelty_tmp'])

    out_df.to_csv(output_csv, index=False)
    print(f'Saved novelty CSV to: {output_csv}')
    print(f'Novelty stats: mean={float(novelty.mean()):.4f}, std={float(novelty.std()):.4f}, '
          f'min={float(novelty.min()):.4f}, max={float(novelty.max()):.4f}')


if __name__ == '__main__':
    main()
