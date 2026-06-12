import torch
import torch.nn as nn

from models import PatchTST
from models.NewsDecayAdapter import DecayAwareResidualAdapter


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.backbone = PatchTST.Model(configs)
        self.news_adapter = DecayAwareResidualAdapter(configs)

    def _normalize(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return x_norm / stdev, means, stdev

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        news_embeddings=None,
        news_time_gaps=None,
        news_mask=None,
        news_novelty=None,
        news_duration_probs=None,
        news_sentiment=None,
    ):
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)

        base_out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        x_norm, _, _ = self._normalize(x_enc)
        return self.news_adapter(
            base_out,
            x_norm,
            news_embeddings=news_embeddings,
            news_time_gaps=news_time_gaps,
            news_mask=news_mask,
            news_novelty=news_novelty,
            news_duration_probs=news_duration_probs,
            news_sentiment=news_sentiment,
        )
