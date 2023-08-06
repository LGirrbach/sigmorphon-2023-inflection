import torch
import torch.nn as nn

from torch import Tensor
from containers import BridgeOutput


class EncoderBridge(nn.Module):
    neg_inf_score = -1e8

    def __init__(
        self, hidden_size: int, num_source_features: int, num_decoder_layers: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_source_features = num_source_features
        self.num_decoder_layers = num_decoder_layers

        self.input2output = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size * self.num_decoder_layers),
            nn.GELU(),
        )

        if self.num_source_features > 0:
            self.input2features = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_source_features), nn.Sigmoid()
            )

    def forward(self, sequences: Tensor, mask: Tensor) -> BridgeOutput:
        # sequences: shape [batch x timesteps x features]
        # mask: shape [batch x timesteps]

        mask = mask.unsqueeze(-1)
        masked_sequences = torch.masked_fill(
            sequences, mask=mask, value=self.neg_inf_score
        )
        max_over_time = torch.max(masked_sequences, dim=1).values
        output = self.input2output(max_over_time)

        if self.num_source_features > 0:
            feature_scores = self.input2features(max_over_time)
        else:
            feature_scores = None

        return BridgeOutput(output=output, feature_scores=feature_scores)
