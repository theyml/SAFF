from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

NEWS_NAME_OVERRIDES = {
    "AAPL": ["apple_title_finbert_emb.csv", "apple_title_finbert_emb_novelty.csv", "apple_title_finbert_emb_duration.csv"],
    "AMD": ["amd_title_finbert_emb.csv"],
    "GOOGL": ["googl_title_finbert_emb.csv"],
    "TSLA": ["tsla_title_finbert_emb.csv"],
}


@dataclass(frozen=True)
class Pair:
    ticker: str
    price_path: Path
    news_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen stock/news pairs by per-prediction-anchor news coverage density. "
            "The anchor rule matches Dataset_Custom: anchor = last input timestamp, "
            "using df_stamp[seq_len - 1 : -pred_len]."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--price-path", type=Path, default=None, help="Single price CSV. Use with --news-path.")
    parser.add_argument("--news-path", type=Path, default=None, help="Single news CSV. Use with --price-path.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker for a single pair. Defaults to price filename prefix.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--news-max-items", type=int, default=32)
    parser.add_argument("--time-col", type=str, default="Date")
    parser.add_argument("--news-time-col", type=str, default="Date")
    parser.add_argument("--freq", choices=["d"], default="d")
    parser.add_argument(
        "--restrict-to-news-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate only anchors between the first and last news date.",
    )
    parser.add_argument("--min-anchors", type=int, default=180)
    parser.add_argument("--min-selected-p10", type=float, default=5.0)
    parser.add_argument("--max-zero-share", type=float, default=0.02)
    parser.add_argument("--max-selected-cv", type=float, default=1.00)
    return parser.parse_args()


def clean_ticker(value: str) -> str:
    return value.strip().upper()


def read_dates(path: Path, column: str) -> pd.Series:
    df = pd.read_csv(path, usecols=[column])
    dates = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if dates.empty:
        return pd.Series(dtype="datetime64[ns]")
    return dates.dt.tz_localize(None).dt.normalize().sort_values().reset_index(drop=True)


def filled_daily_price_dates(price_path: Path, time_col: str) -> pd.Series:
    dates = read_dates(price_path, time_col).drop_duplicates().sort_values().reset_index(drop=True)
    if dates.empty:
        return dates
    full = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    return pd.Series(full, name=time_col)


def anchor_dates(price_dates: pd.Series, seq_len: int, pred_len: int) -> pd.Series:
    end = len(price_dates) - pred_len
    if seq_len - 1 >= end:
        return pd.Series(dtype="datetime64[ns]")
    return price_dates.iloc[seq_len - 1 : end].reset_index(drop=True)


def rolling_window_counts(anchor_values: np.ndarray, news_values: np.ndarray, lookback_days: int) -> np.ndarray:
    low_values = (pd.to_datetime(anchor_values) - pd.Timedelta(days=lookback_days)).astype("datetime64[ns]").astype("int64")
    anchor_ns = pd.to_datetime(anchor_values).astype("datetime64[ns]").astype("int64")
    news_ns = pd.to_datetime(news_values).astype("datetime64[ns]").astype("int64")
    left = np.searchsorted(news_ns, low_values, side="left")
    right = np.searchsorted(news_ns, anchor_ns, side="right")
    return right - left


def pct(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, q))


def summarize_counts(prefix: str, counts: np.ndarray) -> dict[str, float]:
    if counts.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_cv": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_max": math.nan,
        }
    mean = float(np.mean(counts))
    std = float(np.std(counts, ddof=0))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_cv": std / mean if mean > 0 else math.inf,
        f"{prefix}_min": float(np.min(counts)),
        f"{prefix}_p10": pct(counts, 10),
        f"{prefix}_p25": pct(counts, 25),
        f"{prefix}_median": pct(counts, 50),
        f"{prefix}_p75": pct(counts, 75),
        f"{prefix}_p90": pct(counts, 90),
        f"{prefix}_max": float(np.max(counts)),
    }


def infer_news_symbol(news_path: Path) -> str | None:
    try:
        symbols = pd.read_csv(news_path, usecols=["Stock_symbol"], nrows=50)["Stock_symbol"].dropna()
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if symbols.empty:
        return None
    return clean_ticker(str(symbols.iloc[0]))


def build_news_index(data_dir: Path) -> dict[str, list[Path]]:
    news_index: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*title_finbert_emb*.csv")):
        symbol = infer_news_symbol(path)
        if symbol:
            news_index.setdefault(symbol, []).append(path)
    return news_index


def choose_news_path(ticker: str, data_dir: Path, news_index: dict[str, list[Path]]) -> Path | None:
    for file_name in NEWS_NAME_OVERRIDES.get(ticker, []):
        path = data_dir / file_name
        if path.exists():
            return path
    candidates = news_index.get(ticker, [])
    if not candidates:
        lower = ticker.lower()
        candidates = sorted(data_dir.glob(f"{lower}_title_finbert_emb*.csv"))
    plain_name = f"{ticker.lower()}_title_finbert_emb.csv"
    for candidate in candidates:
        if candidate.name == plain_name:
            return candidate
    return candidates[0] if candidates else None


def discover_pairs(data_dir: Path) -> list[Pair]:
    news_index = build_news_index(data_dir)
    pairs: list[Pair] = []
    for price_path in sorted(data_dir.glob("*_news_aligned.csv")):
        ticker = clean_ticker(price_path.name.removesuffix("_news_aligned.csv"))
        news_path = choose_news_path(ticker, data_dir, news_index)
        if news_path is not None:
            pairs.append(Pair(ticker=ticker, price_path=price_path, news_path=news_path))
    return pairs


def screen_pair(pair: Pair, args: argparse.Namespace) -> dict[str, object]:
    price_dates = filled_daily_price_dates(pair.price_path, args.time_col)
    anchors = anchor_dates(price_dates, args.seq_len, args.pred_len)
    news_dates = read_dates(pair.news_path, args.news_time_col)

    if args.restrict_to_news_range and not anchors.empty and not news_dates.empty:
        anchors = anchors[(anchors >= news_dates.min()) & (anchors <= news_dates.max())].reset_index(drop=True)

    raw_counts = rolling_window_counts(anchors.to_numpy(), news_dates.to_numpy(), args.lookback_days)
    selected_counts = np.minimum(raw_counts, args.news_max_items)

    anchor_count = int(len(anchors))
    selected_summary = summarize_counts("selected", selected_counts)
    raw_summary = summarize_counts("raw_window", raw_counts)
    zero_share = float(np.mean(selected_counts == 0)) if anchor_count else math.nan
    full_share = float(np.mean(selected_counts >= args.news_max_items)) if anchor_count else math.nan
    under_min_share = float(np.mean(selected_counts < args.min_selected_p10)) if anchor_count else math.nan

    selected_p10 = selected_summary["selected_p10"]
    selected_cv = selected_summary["selected_cv"]
    passes = (
        anchor_count >= args.min_anchors
        and selected_p10 >= args.min_selected_p10
        and zero_share <= args.max_zero_share
        and selected_cv <= args.max_selected_cv
    )
    score = (
        min(selected_summary["selected_mean"] / args.news_max_items, 1.0)
        * (1.0 - min(zero_share, 1.0))
        * (1.0 / (1.0 + max(selected_cv, 0.0)))
    )

    return {
        "pass_stable_coverage": passes,
        "coverage_score": score,
        "ticker": pair.ticker,
        "anchor_count": anchor_count,
        "anchor_start": anchors.min().date().isoformat() if anchor_count else None,
        "anchor_end": anchors.max().date().isoformat() if anchor_count else None,
        "news_rows": int(len(news_dates)),
        "news_start": news_dates.min().date().isoformat() if len(news_dates) else None,
        "news_end": news_dates.max().date().isoformat() if len(news_dates) else None,
        "zero_selected_share": zero_share,
        "under_min_selected_share": under_min_share,
        "full_selected_share": full_share,
        **selected_summary,
        **raw_summary,
        "price_path": str(pair.price_path),
        "news_path": str(pair.news_path),
    }


def main() -> None:
    args = parse_args()
    if bool(args.price_path) != bool(args.news_path):
        raise SystemExit("--price-path and --news-path must be provided together.")

    if args.price_path and args.news_path:
        ticker = clean_ticker(args.ticker or args.price_path.stem.removesuffix("_news_aligned"))
        pairs = [Pair(ticker=ticker, price_path=args.price_path, news_path=args.news_path)]
    else:
        pairs = discover_pairs(args.data_dir)

    if not pairs:
        raise SystemExit(f"No price/news pairs found under {args.data_dir}")

    rows = [screen_pair(pair, args) for pair in pairs]
    df = pd.DataFrame(rows).sort_values(
        by=["pass_stable_coverage", "coverage_score", "selected_p10", "zero_selected_share"],
        ascending=[False, False, False, True],
    )

    output_csv = args.output_csv or (args.data_dir / f"news_anchor_coverage_screen_seq{args.seq_len}_pred{args.pred_len}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    display_cols = [
        "pass_stable_coverage",
        "coverage_score",
        "ticker",
        "anchor_count",
        "anchor_start",
        "anchor_end",
        "news_rows",
        "selected_mean",
        "selected_p10",
        "selected_cv",
        "zero_selected_share",
        "full_selected_share",
        "raw_window_mean",
        "raw_window_p10",
        "news_path",
    ]
    print(f"[output] {output_csv}")
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
