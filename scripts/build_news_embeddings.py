import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# [short, long, unsure]
DEFAULT_EVENT_DURATION_PRIORS = {
    'Merger/Acquisition': [0.10, 0.80, 0.10],
    'Investment': [0.15, 0.75, 0.10],
    'Facility': [0.15, 0.75, 0.10],
    'Financing': [0.20, 0.65, 0.15],
    'Product/Service': [0.20, 0.65, 0.15],
    'Macroeconomics': [0.15, 0.70, 0.15],
    'Deal': [0.25, 0.55, 0.20],
    'Legal': [0.20, 0.60, 0.20],
    'Employment': [0.25, 0.45, 0.30],
    'CSR/Brand': [0.25, 0.45, 0.30],
    'FinancialReport': [0.30, 0.45, 0.25],
    'Revenue': [0.35, 0.40, 0.25],
    'Profit/Loss': [0.35, 0.40, 0.25],
    'Expense': [0.35, 0.35, 0.30],
    'Dividend': [0.45, 0.30, 0.25],
    'Rating': [0.70, 0.10, 0.20],
    'SalesVolume': [0.75, 0.10, 0.15],
    'SecurityValue': [0.70, 0.10, 0.20],
}

DEFAULT_ZS_LABELS = [
    'short-lived market impact',
    'long-lasting market impact',
    'uncertain impact duration',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build offline news embeddings and optional duration probabilities for Financial-Time-Series.'
    )
    parser.add_argument('--input_csv', type=str, required=True, help='Input news CSV.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV.')

    # Embedding options.
    parser.add_argument('--model_name_or_path', type=str, default=None, help='HF model name or local path for embeddings.')
    parser.add_argument('--text_cols', nargs='+', default=None, help='Columns to concatenate as encoder input.')
    parser.add_argument('--skip_embeddings', action='store_true', help='Skip embedding generation and only append duration columns.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--prefix', type=str, default='news_emb_')
    parser.add_argument('--device', type=str, default=None, help='cuda / cpu. Default: auto-detect.')
    parser.add_argument('--normalize', action='store_true', help='L2-normalize embeddings.')

    # Duration options.
    parser.add_argument('--build_duration', action='store_true', help='Append duration probability columns.')
    parser.add_argument(
        '--duration_text_cols',
        nargs='+',
        default=None,
        help='Columns used for duration estimation. Default: same as --text_cols.',
    )
    parser.add_argument(
        '--duration_event_model_name_or_path',
        type=str,
        default='ritessshhh/FinDeBERTa',
        help='HF model name or local path for finance event classification.',
    )
    parser.add_argument(
        '--duration_zs_model_name_or_path',
        type=str,
        default='MoritzLaurer/ModernBERT-large-zeroshot-v2.0',
        help='HF model name or local path for zero-shot duration classification.',
    )
    parser.add_argument(
        '--duration_zs_onnx_model_path',
        type=str,
        default=None,
        help='Optional local ONNX model path for zero-shot duration classification.',
    )
    parser.add_argument(
        '--duration_skip_event_model',
        action='store_true',
        help='Skip the finance event expert and use only zero-shot duration classification.',
    )
    parser.add_argument('--duration_event_top_k', type=int, default=5, help='Top-k event labels used to build the event prior.')
    parser.add_argument('--duration_event_weight', type=float, default=0.4, help='Weight of the event prior expert in fused duration scores.')
    parser.add_argument('--duration_zs_weight', type=float, default=0.6, help='Weight of the zero-shot expert in fused duration scores.')
    parser.add_argument(
        '--duration_hypothesis_template',
        type=str,
        default='The stock-market impact duration of this financial news is {}.',
        help='Hypothesis template for zero-shot classification.',
    )
    parser.add_argument(
        '--duration_output_prefix',
        type=str,
        default='duration_',
        help='Prefix for duration output columns.',
    )
    parser.add_argument(
        '--duration_source_col',
        type=str,
        default='duration_source',
        help='Column name recording how the duration probabilities were produced.',
    )
    parser.add_argument(
        '--duration_event_entropy_weight',
        type=float,
        default=0.0,
        help='Entropy weight applied to the event expert when boosting unsure. Default 0 to avoid systematic unsure bias.',
    )
    parser.add_argument(
        '--duration_zs_entropy_weight',
        type=float,
        default=0.0,
        help='Entropy weight applied to the zero-shot expert when boosting unsure. Default 0 to avoid systematic unsure bias.',
    )
    parser.add_argument(
        '--duration_margin_weight',
        type=float,
        default=0.0,
        help='Margin-based uncertainty weight from the zero-shot expert. Default 0 to avoid systematic unsure bias.',
    )
    parser.add_argument(
        '--duration_unsure_boost_cap',
        type=float,
        default=0.15,
        help='Upper bound for any extra unsure-logit boost when uncertainty boosting is enabled.',
    )
    parser.add_argument(
        '--duration_temperature',
        type=float,
        default=1.0,
        help='Temperature for the final fused duration distribution. Values >1 flatten the distribution.',
    )
    return parser.parse_args()


def resolve_device(device_arg):
    return device_arg or ('cuda' if torch.cuda.is_available() else 'cpu')


def is_local_path(path_or_repo):
    if path_or_repo is None:
        return False
    return os.path.exists(path_or_repo) or path_or_repo.startswith('/')


def hf_device_index(device):
    if not device.startswith('cuda'):
        return -1
    if ':' in device:
        return int(device.split(':', 1)[1])
    return 0


def build_texts(df: pd.DataFrame, text_cols):
    if not text_cols:
        raise ValueError('text_cols must be provided.')

    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing text columns: {missing}')

    text_parts = []
    for col in text_cols:
        text_parts.append(df[col].fillna('').astype(str).str.strip())

    texts = text_parts[0]
    for part in text_parts[1:]:
        texts = texts + ' ' + part

    texts = texts.str.replace(r'\s+', ' ', regex=True).str.strip()
    return texts.tolist()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 1e-8, None)
    x = x / x.sum(axis=1, keepdims=True)
    return x


def entropy_rows(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-8, 1.0)
    return -(x * np.log(x)).sum(axis=1)


def softmax_np(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def print_prob_summary(name, probs):
    probs = np.asarray(probs, dtype=np.float32)
    mean_probs = probs.mean(axis=0)
    print(
        f'{name} mean probs: '
        f'short={mean_probs[0]:.4f}, '
        f'long={mean_probs[1]:.4f}, '
        f'unsure={mean_probs[2]:.4f}'
    )


def encode_embeddings(texts, args, device):
    local_only = is_local_path(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=local_only)
    model = AutoModel.from_pretrained(args.model_name_or_path, local_files_only=local_only)
    model.to(device)
    model.eval()

    all_embeddings = []

    for start in tqdm(range(0, len(texts), args.batch_size), desc='Encoding news', unit='batch'):
        batch_texts = texts[start:start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = mean_pool(outputs.last_hidden_state, encoded['attention_mask'])
            if args.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.detach().cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    emb_dim = all_embeddings.shape[1]
    print(f'Embedding dim: {emb_dim}')
    return all_embeddings


def get_model_label_names(model):
    id2label = getattr(model.config, 'id2label', None) or {}
    return [id2label[idx] for idx in sorted(id2label.keys())]


def build_event_prior_from_logits(texts, args, device):
    local_only = is_local_path(args.duration_event_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.duration_event_model_name_or_path, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.duration_event_model_name_or_path,
        local_files_only=local_only,
    )
    model.to(device)
    model.eval()

    label_names = get_model_label_names(model)
    if not label_names:
        raise ValueError('Event classifier does not expose id2label; cannot map events to duration priors.')

    event_probs = []
    event_prior = []
    event_top_labels = []

    for start in tqdm(range(0, len(texts), args.batch_size), desc='Scoring event labels', unit='batch'):
        batch_texts = texts[start:start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt',
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        event_probs.append(probs)

    event_probs = np.concatenate(event_probs, axis=0)
    top_k = max(1, min(args.duration_event_top_k, event_probs.shape[1]))

    default_prior = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    for row in event_probs:
        top_idx = np.argsort(row)[::-1][:top_k]
        top_weights = row[top_idx]
        top_weights = top_weights / np.clip(top_weights.sum(), 1e-8, None)

        prior = np.zeros(3, dtype=np.float32)
        top_names = []
        for idx, weight in zip(top_idx, top_weights):
            label_name = label_names[idx]
            top_names.append(label_name)
            mapped = DEFAULT_EVENT_DURATION_PRIORS.get(label_name, default_prior)
            prior += float(weight) * np.asarray(mapped, dtype=np.float32)

        event_prior.append(prior)
        event_top_labels.append('|'.join(top_names))

    event_prior = normalize_rows(np.asarray(event_prior, dtype=np.float32))
    return event_prior, event_top_labels


def build_zero_shot_duration_probs(texts, args, device):
    local_only = is_local_path(args.duration_zs_model_name_or_path)
    zs_pipe = pipeline(
        task='zero-shot-classification',
        model=args.duration_zs_model_name_or_path,
        device=hf_device_index(device),
        local_files_only=local_only,
    )

    probs = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc='Zero-shot duration', unit='batch'):
        batch_texts = texts[start:start + args.batch_size]
        outputs = zs_pipe(
            batch_texts,
            candidate_labels=DEFAULT_ZS_LABELS,
            hypothesis_template=args.duration_hypothesis_template,
            multi_label=False,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for out in outputs:
            label_to_score = dict(zip(out['labels'], out['scores']))
            probs.append([
                float(label_to_score.get(DEFAULT_ZS_LABELS[0], 0.0)),
                float(label_to_score.get(DEFAULT_ZS_LABELS[1], 0.0)),
                float(label_to_score.get(DEFAULT_ZS_LABELS[2], 0.0)),
            ])

    return normalize_rows(np.asarray(probs, dtype=np.float32))


def build_zero_shot_duration_probs_onnx(texts, args):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            'onnxruntime is required when --duration_zs_onnx_model_path is used. '
            'Install it in your environment first.'
        ) from exc

    onnx_model_path = args.duration_zs_onnx_model_path
    if not onnx_model_path or not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f'ONNX zero-shot model not found: {onnx_model_path}')

    tokenizer_path = args.duration_zs_model_name_or_path
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            'A local tokenizer/model directory must be provided via --duration_zs_model_name_or_path '
            'when using --duration_zs_onnx_model_path.'
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    config_path = Path(tokenizer_path) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found under tokenizer path: {tokenizer_path}')

    config = json_load(config_path)
    label2id = config.get('label2id', {})
    entailment_idx = label2id.get('entailment', 0)

    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_names = {inp.name for inp in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    probs = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc='Zero-shot duration (onnx)', unit='batch'):
        batch_texts = texts[start:start + args.batch_size]
        paired_premises = []
        paired_hypotheses = []
        for text in batch_texts:
            for label in DEFAULT_ZS_LABELS:
                paired_premises.append(text)
                paired_hypotheses.append(args.duration_hypothesis_template.format(label))

        encoded = tokenizer(
            paired_premises,
            paired_hypotheses,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='np',
        )

        ort_inputs = {}
        for key, value in encoded.items():
            if key in input_names:
                ort_inputs[key] = value.astype(np.int64, copy=False)

        logits = session.run([output_name], ort_inputs)[0]
        if logits.ndim != 2:
            raise ValueError(f'Unexpected ONNX logits shape: {logits.shape}')

        label_probs = softmax_np(logits.astype(np.float32))
        entailment_scores = label_probs[:, entailment_idx]
        batch_probs = entailment_scores.reshape(len(batch_texts), len(DEFAULT_ZS_LABELS))
        probs.append(batch_probs)

    return normalize_rows(np.concatenate(probs, axis=0).astype(np.float32))


def fuse_duration_experts(event_prior, zs_probs, args):
    event_weight = float(args.duration_event_weight)
    zs_weight = float(args.duration_zs_weight)
    total_weight = max(event_weight + zs_weight, 1e-8)
    event_weight /= total_weight
    zs_weight /= total_weight

    log_event = np.log(np.clip(event_prior, 1e-8, 1.0))
    log_zs = np.log(np.clip(zs_probs, 1e-8, 1.0))
    fused_logits = event_weight * log_event + zs_weight * log_zs

    event_entropy = entropy_rows(event_prior) / math.log(3.0)
    zs_entropy = entropy_rows(zs_probs) / math.log(3.0)
    zs_margin = np.abs(zs_probs[:, 0] - zs_probs[:, 1])
    uncertainty_boost = (
        float(args.duration_event_entropy_weight) * event_entropy
        + float(args.duration_zs_entropy_weight) * zs_entropy
        + float(args.duration_margin_weight) * (1.0 - zs_margin)
    )

    uncertainty_boost = np.clip(
        uncertainty_boost,
        0.0,
        max(float(args.duration_unsure_boost_cap), 0.0),
    )
    fused_logits[:, 2] += uncertainty_boost
    fused_logits /= max(float(args.duration_temperature), 1e-6)
    fused_probs = softmax_np(fused_logits)
    return fused_probs


def append_duration_columns(df, texts, args, device):
    print('Building duration probabilities...')
    if args.duration_skip_event_model:
        event_prior = None
        event_top_labels = None
    else:
        event_prior, event_top_labels = build_event_prior_from_logits(texts, args, device)

    if args.duration_zs_onnx_model_path:
        zs_probs = build_zero_shot_duration_probs_onnx(texts, args)
        zs_source = 'zs_onnx'
    else:
        zs_probs = build_zero_shot_duration_probs(texts, args, device)
        zs_source = 'zs_transformers'

    if event_prior is None:
        fused_probs = zs_probs
    else:
        fused_probs = fuse_duration_experts(event_prior, zs_probs, args)

    if event_prior is not None:
        print_prob_summary('Event prior', event_prior)
    print_prob_summary('Zero-shot', zs_probs)
    print_prob_summary('Fused duration', fused_probs)

    prefix = args.duration_output_prefix
    out_df = df.copy()
    out_df[f'{prefix}short_prob'] = fused_probs[:, 0]
    out_df[f'{prefix}long_prob'] = fused_probs[:, 1]
    out_df[f'{prefix}unsure_prob'] = fused_probs[:, 2]
    if event_prior is not None:
        out_df[f'{prefix}event_short_prob'] = event_prior[:, 0]
        out_df[f'{prefix}event_long_prob'] = event_prior[:, 1]
        out_df[f'{prefix}event_unsure_prob'] = event_prior[:, 2]
        out_df[f'{prefix}event_top_labels'] = event_top_labels
    out_df[f'{prefix}zs_short_prob'] = zs_probs[:, 0]
    out_df[f'{prefix}zs_long_prob'] = zs_probs[:, 1]
    out_df[f'{prefix}zs_unsure_prob'] = zs_probs[:, 2]
    out_df[args.duration_source_col] = zs_source if event_prior is None else f'event+{zs_source}'
    return out_df


def json_load(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f'Using device: {device}')

    df = pd.read_csv(input_csv)
    print(f'Loaded {len(df)} rows from {input_csv}')

    out_df = df.copy()

    embedding_texts = None
    if not args.skip_embeddings:
        if not args.model_name_or_path:
            raise ValueError('--model_name_or_path is required unless --skip_embeddings is set.')
        if not args.text_cols:
            raise ValueError('--text_cols is required unless --skip_embeddings is set.')

        embedding_texts = build_texts(df, args.text_cols)
        embeddings = encode_embeddings(embedding_texts, args, device)
        emb_cols = [f'{args.prefix}{i}' for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        out_df = pd.concat([out_df.reset_index(drop=True), emb_df], axis=1)

    if args.build_duration:
        duration_text_cols = args.duration_text_cols or args.text_cols
        if not duration_text_cols:
            raise ValueError('--duration_text_cols must be provided when building duration without --text_cols.')
        duration_texts = build_texts(df, duration_text_cols)
        out_df = append_duration_columns(out_df, duration_texts, args, device)

    out_df.to_csv(output_csv, index=False)
    print(f'Saved: {output_csv}')


if __name__ == '__main__':
    main()
