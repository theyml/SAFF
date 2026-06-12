"""
Incremental Phase-1 entrypoint for news-aware forecasting experiments.

This file intentionally leaves the original `run.py` unchanged.
It reuses the existing training flow and only extends the parser with
news-related arguments needed by the decay-aware prototype.
"""

from run import get_basic_parser, main
from exp.exp_basic import Exp_Basic


def add_news_decay_arguments(parser):
    # Toggle for switching the data/model pipeline to the news-aware variant.
    parser.add_argument('--use_news', action='store_true',
                        help='Enable the finance-news dataset branch and pass news tensors to the model.')

    # News source configuration.
    parser.add_argument('--news_path', type=str, default=None,
                        help='News CSV file placed under root_path or specified as an absolute path.')
    parser.add_argument('--news_time_col', type=str, default='Date',
                        help='Timestamp column in the news CSV.')
    parser.add_argument('--news_id_col', type=str, default=None,
                        help='Optional unique identifier column for news rows.')
    parser.add_argument('--news_text_col', type=str, default=None,
                        help='Optional raw text column. Phase 1 does not encode raw text online, but keeps the schema explicit.')
    parser.add_argument('--news_title_col', type=str, default=None,
                        help='Optional headline/title column. Phase 1 treats precomputed embeddings as the main signal.')

    # Phase-1 expects precomputed numeric news embeddings in the CSV.
    parser.add_argument('--news_embed_prefix', type=str, default='news_emb_',
                        help='Prefix for numeric news embedding columns.')
    parser.add_argument('--news_emb_dim', type=int, default=32,
                        help='Expected news embedding width consumed by iTransformerNewsDecay.')

    # Candidate-news memory controls.
    parser.add_argument('--news_max_items', type=int, default=32,
                        help='Maximum number of historical news items kept per forecast anchor.')
    parser.add_argument('--news_lookback_days', type=int, default=30,
                        help='Long candidate lookback horizon in days before time-decay reweighting.')
    parser.add_argument('--news_selection_mode', type=str, default='latest',
                        choices=['latest', 'quality', 'quality_recency'],
                        help='How to choose up to news_max_items from the lookback window. '
                             'latest preserves the old behavior; quality uses news_rank_col; '
                             'quality_recency combines news_rank_col with recency.')
    parser.add_argument('--news_rank_col', type=str, default=None,
                        help='Optional numeric quality/relevance column used by quality-aware news selection, '
                             'for example zs_rel_weight.')
    parser.add_argument('--news_rank_recency_weight', type=float, default=0.15,
                        help='Weight of the normalized recency bonus in quality_recency selection.')
    parser.add_argument('--news_rank_fill_value', type=float, default=0.0,
                        help='Fill value for missing/non-numeric news_rank_col values.')

    # Optional novelty signal.
    parser.add_argument('--use_novelty', action='store_true',
                        help='Enable novelty gating if a novelty column is available.')
    parser.add_argument('--news_novelty_col', type=str, default=None,
                        help='Optional novelty score column in the news CSV.')
    parser.add_argument('--use_sentiment', action='store_true',
                        help='Enable an explicit headline-sentiment scalar in the news adapter.')
    parser.add_argument('--news_sentiment_col', type=str, default=None,
                        help='Optional numeric headline sentiment column. If missing, a lightweight title lexicon is used.')
    parser.add_argument('--news_sentiment_title_col', type=str, default='Article_title',
                        help='Headline/title column used by the fallback sentiment lexicon.')
    parser.add_argument('--use_signed_impact', action='store_true',
                        help='Use a learned sentiment-conditioned signed impact gate before news aggregation.')
    parser.add_argument('--use_signed_decay_kernel', action='store_true',
                        help='Use a learned signed temporal decay kernel instead of normalized positive attention weights.')
    parser.add_argument('--news_duration_prefix', type=str, default='duration_',
                        help='Prefix for offline duration probability columns in the news CSV.')
    parser.add_argument('--news_duration_short_col', type=str, default=None,
                        help='Optional short-duration probability column. Defaults to <news_duration_prefix>short_prob.')
    parser.add_argument('--news_duration_long_col', type=str, default=None,
                        help='Optional long-duration probability column. Defaults to <news_duration_prefix>long_prob.')
    parser.add_argument('--news_duration_unsure_col', type=str, default=None,
                        help='Optional unsure-duration probability column. Defaults to <news_duration_prefix>unsure_prob.')

    # Small decay module controls.
    parser.add_argument('--news_decay_hidden_dim', type=int, default=64,
                        help='Hidden width of the text-conditioned decay-rate MLP.')
    parser.add_argument('--news_residual_scale', type=float, default=1.0,
                        help='Scalar multiplier applied to the final news residual before adding it to the price forecast.')
    parser.add_argument('--disable_time_decay', action='store_true',
                        help='Disable time decay so news only affects the immediate next prediction step.')
    parser.add_argument('--use_market_state_decay', action='store_true',
                        help='Condition news decay on the current market state from the price encoder.')
    parser.add_argument('--use_news_selector', action='store_true',
                        help='Select relevant news with a state-aware relevance gate before decay aggregation.')
    parser.add_argument('--use_channel_specific_news', action='store_true',
                        help='Build variable-specific news contexts so channels like Close and Volume can react differently.')
    parser.add_argument('--use_novelty_persistence', action='store_true',
                        help='Slow down decay for novel news via a learned persistence gate.')
    parser.add_argument('--use_duration_persistence', action='store_true',
                        help='Use duration-aware alpha control from offline short/long/unsure probabilities when available; otherwise fall back to a learned 3-class controller.')
    parser.add_argument('--duration_persistence_values', type=str, default='0.75,1.25,1.0',
                        help='Comma-separated alpha-control multipliers for short,long,unsure duration classes.')
    parser.add_argument('--duration_confidence_threshold', type=float, default=0.55,
                        help='Apply duration control only when max(short,long) reaches this confidence threshold.')
    parser.add_argument('--duration_margin_threshold', type=float, default=0.15,
                        help='Apply duration control only when |short-long| reaches this margin threshold.')
    parser.add_argument('--debug_news_stats', action='store_true',
                        help='Print internal news-branch statistics during training to verify advanced modules are active.')
    parser.add_argument('--debug_news_stats_every', type=int, default=200,
                        help='Print news debug statistics every N train batches when --debug_news_stats is enabled.')
    parser.add_argument('--debug_news_stats_in_val', action='store_true',
                        help='Also print news debug statistics during validation/test when --debug_news_stats is enabled.')

    return parser


if __name__ == '__main__':
    parser = get_basic_parser(non_stationary=True, timemixer=True)
    parser.add_argument('--model', type=str, default='iTransformerNewsDecay',
                        choices=list(Exp_Basic.model_dict.keys()),
                        help='Model name. Phase 1 adds iTransformerNewsDecay without modifying original iTransformer.')
    parser = add_news_decay_arguments(parser)
    args = parser.parse_args()
    main(args)
