"""
Entry point for Transformer-only long-news cross-attention experiments.

This keeps the original `decay_run.py` intact while reusing its news-data
arguments. The new model is registered as `TransformerCrossAttentionNewsDecay`.
"""

from decay_run import add_news_decay_arguments
from exp.exp_basic import Exp_Basic
from run import get_basic_parser, main


def add_cross_attention_arguments(parser):
    parser.add_argument(
        '--news_cross_use_fixed_decay',
        action='store_true',
        help='Use a fixed temporal decay rate in cross-attention instead of a learned decay MLP.',
    )
    parser.add_argument(
        '--news_cross_fixed_decay_alpha',
        type=float,
        default=0.05,
        help='Fixed alpha used when --news_cross_use_fixed_decay is enabled.',
    )
    parser.add_argument(
        '--news_cross_gap_scale',
        type=float,
        default=1.0,
        help='Scale applied to effective news gaps before the cross-attention temporal bias.',
    )
    return parser


if __name__ == '__main__':
    parser = get_basic_parser(non_stationary=True, timemixer=True)
    parser.add_argument(
        '--model',
        type=str,
        default='TransformerCrossAttentionNewsDecay',
        choices=list(Exp_Basic.model_dict.keys()),
        help='Model name. This entry point is intended for Transformer news cross-attention ablations.',
    )
    parser = add_news_decay_arguments(parser)
    parser = add_cross_attention_arguments(parser)
    args = parser.parse_args()
    main(args)
