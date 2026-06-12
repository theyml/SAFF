import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


TASKS = {
    'specificity': {
        'labels': [
            'target company specific news',
            'target product or business specific news',
            'sector, supply chain, or competitor context',
            'generic market, macro, ETF flow, or stock-list news',
            'unrelated or wrong company news',
        ],
        'cols': [
            'event_spec_company_prob',
            'event_spec_product_business_prob',
            'event_spec_sector_competitor_prob',
            'event_spec_market_macro_prob',
            'event_spec_unrelated_prob',
        ],
        'prompt': 'Classify how specifically this news is connected to the target company stock price.',
    },
    'importance': {
        'labels': [
            'high-impact tradable event for the target stock',
            'moderate business-relevant event for the target stock',
            'low-impact routine update for the target stock',
            'broad market or sector background with weak firm-specific impact',
            'unrelated or non-informative noise for the target stock',
        ],
        'cols': [
            'event_imp_high_prob',
            'event_imp_moderate_prob',
            'event_imp_low_prob',
            'event_imp_background_prob',
            'event_imp_noise_prob',
        ],
        'prompt': 'Classify the likely importance of this news for forecasting the target company stock price.',
    },
    'direction': {
        'labels': [
            'clearly positive for the target stock price',
            'clearly negative for the target stock price',
            'mixed or uncertain direction for the target stock price',
            'neutral or no meaningful price impact',
            'unrelated or wrong company news',
        ],
        'cols': [
            'event_dir_positive_prob',
            'event_dir_negative_prob',
            'event_dir_mixed_prob',
            'event_dir_neutral_prob',
            'event_dir_unrelated_prob',
        ],
        'prompt': 'Classify the likely directional impact of this news on the target company stock price.',
    },
    'horizon': {
        'labels': [
            'very short-lived impact within one trading day',
            'short-horizon impact over several trading days',
            'medium-horizon impact over several weeks',
            'long-horizon fundamental impact over months',
            'unclear or no forecastable impact horizon',
        ],
        'cols': [
            'event_horizon_intraday_prob',
            'event_horizon_short_prob',
            'event_horizon_medium_prob',
            'event_horizon_long_prob',
            'event_horizon_unclear_prob',
        ],
        'prompt': 'Classify the expected persistence horizon of this news impact on the target company stock price.',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append event-structured zero-shot scores for news selection ablations.'
    )
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
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
    parser.add_argument('--hypothesis_template', default='This text is {}.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', default=None, help='cuda / cpu. Default: auto-detect.')
    parser.add_argument(
        '--torch_dtype',
        default='auto',
        choices=['auto', 'float32', 'float16', 'bfloat16'],
        help='Inference dtype. Use float16 or bfloat16 on A100 to reduce memory.',
    )
    parser.add_argument(
        '--disable_torch_compile',
        action='store_true',
        help='Disable torch.compile paths where supported; useful for ModernBERT memory stability.',
    )
    parser.add_argument(
        '--tasks',
        default='specificity,importance,direction,horizon',
        help='Comma-separated subset of: specificity,importance,direction,horizon.',
    )
    parser.add_argument('--sector_weight', type=float, default=0.35)
    parser.add_argument('--market_macro_weight', type=float, default=0.05)
    parser.add_argument('--importance_moderate_weight', type=float, default=0.50)
    parser.add_argument('--importance_low_weight', type=float, default=0.10)
    parser.add_argument('--rank_eps', type=float, default=1e-4)
    return parser.parse_args()


def is_local_path(model_name_or_path):
    return os.path.exists(model_name_or_path) or str(model_name_or_path).startswith(('.', '/', '~'))


def resolve_device(device):
    if device:
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def resolve_torch_dtype(dtype):
    import torch

    if dtype == 'float16':
        return torch.float16
    if dtype == 'bfloat16':
        return torch.bfloat16
    if dtype == 'float32':
        return torch.float32
    return None


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    row_sum = values.sum(axis=1, keepdims=True)
    return values / np.clip(row_sum, 1e-8, None)


def build_premises(df, args, task_prompt):
    texts = df[args.text_col].fillna('').astype(str).tolist()
    prefix = (
        f'Target company: {args.company_name}\n'
        f'Ticker: {args.ticker}\n'
        f'Sector: {args.sector}\n'
        f'Business description: {args.business_description}\n\n'
        'News text:\n'
    )
    suffix = f'\n\nTask: {task_prompt}'
    return [prefix + text + suffix for text in texts]


def load_zero_shot_model(args, device):
    if args.disable_torch_compile:
        os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
        os.environ.setdefault('DISABLE_TORCH_COMPILE', '1')

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_only = is_local_path(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=local_only)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    model_kwargs = {'local_files_only': local_only}
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    ).to(device)
    if args.disable_torch_compile and hasattr(model.config, 'reference_compile'):
        model.config.reference_compile = False
    model.eval()

    label2id = {str(k).lower(): int(v) for k, v in getattr(model.config, 'label2id', {}).items()}
    entailment_idx = label2id.get('entailment')
    if entailment_idx is None:
        entailment_idx = label2id.get('entailment_label')
    if entailment_idx is None:
        entailment_idx = int(getattr(model.config, 'num_labels', 3)) - 1
    return tokenizer, model, entailment_idx


def zero_shot_probs(premises, labels, args, tokenizer, model, entailment_idx, device, desc):
    import torch
    import torch.nn.functional as F

    hypotheses = [args.hypothesis_template.format(label) for label in labels]
    probs = []
    for start in tqdm(range(0, len(premises), args.batch_size), desc=desc, unit='batch'):
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
        with torch.inference_mode():
            logits = model(**encoded).logits
            label_probs = F.softmax(logits, dim=-1)
            entailment_scores = label_probs[:, entailment_idx]
            batch_probs = entailment_scores.reshape(len(batch), len(labels))
            probs.append(batch_probs.detach().cpu().numpy())

    return normalize_rows(np.concatenate(probs, axis=0))


def selected_tasks(args):
    tasks = [task.strip() for task in args.tasks.split(',') if task.strip()]
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f'Unknown tasks: {unknown}. Valid tasks: {sorted(TASKS)}')
    return tasks


def append_task_columns(out, task, probs):
    spec = TASKS[task]
    for col, values in zip(spec['cols'], probs.T):
        out[col] = values
    out[f'event_{task}_label'] = [spec['labels'][int(i)] for i in probs.argmax(axis=1)]
    out[f'event_{task}_confidence'] = probs.max(axis=1)


def append_composite_scores(out, args):
    def col(name, default=0.0):
        if name in out.columns:
            return pd.to_numeric(out[name], errors='coerce').fillna(default).astype(float)
        return pd.Series(default, index=out.index, dtype=float)

    spec_noise = col('event_spec_market_macro_prob') + col('event_spec_unrelated_prob')
    out['event_firm_specificity'] = np.clip(
        col('event_spec_company_prob')
        + col('event_spec_product_business_prob')
        + float(args.sector_weight) * col('event_spec_sector_competitor_prob')
        + float(args.market_macro_weight) * col('event_spec_market_macro_prob'),
        0.0,
        1.0,
    )
    out['event_importance'] = np.clip(
        col('event_imp_high_prob')
        + float(args.importance_moderate_weight) * col('event_imp_moderate_prob')
        + float(args.importance_low_weight) * col('event_imp_low_prob'),
        0.0,
        1.0,
    )
    out['event_direction_confidence'] = np.maximum(
        col('event_dir_positive_prob').to_numpy(),
        col('event_dir_negative_prob').to_numpy(),
    )
    out['event_direction_signed'] = col('event_dir_positive_prob') - col('event_dir_negative_prob')
    out['event_noise_prob'] = np.clip(
        0.5 * spec_noise
        + 0.3 * col('event_imp_noise_prob')
        + 0.2 * col('event_dir_unrelated_prob'),
        0.0,
        1.0,
    )
    out['event_rank_score'] = (
        (out['event_firm_specificity'] + float(args.rank_eps))
        * (out['event_importance'] + float(args.rank_eps))
        * (out['event_direction_confidence'] + float(args.rank_eps))
        * (1.0 - out['event_noise_prob'])
    )
    out['event_rank_score_no_direction'] = (
        (out['event_firm_specificity'] + float(args.rank_eps))
        * (out['event_importance'] + float(args.rank_eps))
        * (1.0 - out['event_noise_prob'])
    )
    out['event_rank_source'] = 'zero_shot_event_scores'
    return out


def print_summary(out, tasks):
    print(f'total_rows={len(out)}')
    for task in tasks:
        label_col = f'event_{task}_label'
        if label_col in out.columns:
            print(f'{label_col}_counts=')
            print(out[label_col].value_counts(dropna=False).to_string())
    for col in [
        'event_firm_specificity',
        'event_importance',
        'event_direction_confidence',
        'event_noise_prob',
        'event_rank_score',
        'event_rank_score_no_direction',
    ]:
        if col in out.columns:
            print(f'{col}_mean={out[col].mean():.4f}')
            print(f'{col}_p90={out[col].quantile(0.90):.4f}')


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    if args.text_col not in df.columns:
        raise ValueError(f'text_col={args.text_col!r} not found in {input_csv}')

    tasks = selected_tasks(args)
    device = resolve_device(args.device)
    tokenizer, model, entailment_idx = load_zero_shot_model(args, device)

    out = df.copy()
    for task in tasks:
        spec = TASKS[task]
        premises = build_premises(df, args, spec['prompt'])
        probs = zero_shot_probs(
            premises,
            spec['labels'],
            args,
            tokenizer,
            model,
            entailment_idx,
            device,
            desc=f'Zero-shot {task}',
        )
        append_task_columns(out, task, probs)

    out = append_composite_scores(out, args)
    print_summary(out, tasks)
    out.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
