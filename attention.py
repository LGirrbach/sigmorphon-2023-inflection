import torch

from torch import Tensor
from typing import Optional
from utils import discretize_sigmoid
from utils import discretize_softmax
from containers import AttentionOutput


def get_hard_attention_scores(
    normalised_attention_scores: Tensor,
    raw_attention_scores: Tensor,
    normalisation: str,
    deterministic_discretize: bool,
) -> Tensor:
    if normalisation == "softmax":
        return discretize_softmax(
            raw_attention_scores, deterministic=deterministic_discretize
        )

    elif normalisation == "sigmoid":
        return discretize_sigmoid(
            normalised_attention_scores, deterministic=deterministic_discretize
        )

    else:
        raise ValueError(
            f"Expected `normalisation` in ['softmax', 'sigmoid'], but found {normalisation}"
        )


def attention(
    encoder_states: Tensor,
    decoder_states: Tensor,
    attention_mask: Tensor,
    values: Optional[Tensor] = None,
    normalisation: str = "softmax",
    hard: bool = True,
    deterministic_discretize: bool = True,
):
    # Initialise Normaliser
    bad_normaliser_error_msg = (
        f"Expected `normalisation` in ['softmax', 'sigmoid'], but found {normalisation}"
    )
    assert normalisation in ["softmax", "sigmoid"], bad_normaliser_error_msg

    # encoder_states: shape [batch x timesteps encoder x features]
    # decoder_states: shape [batch x timesteps decoder x features]
    batch, encoder_timesteps, _ = encoder_states.shape
    _, decoder_timesteps, _ = decoder_states.shape

    # Set values
    if values is None:
        values = encoder_states
    else:
        assert isinstance(values, Tensor)
        assert len(values.shape) == 3
        assert values.shape[1] == encoder_states.shape[1]

    # Compute raw attention scores
    attention_scores_unmasked = torch.bmm(
        encoder_states, decoder_states.transpose(1, 2)
    )
    # attention_scores_unmasked: shape [batch x timesteps encoder x timesteps decoder]

    # Mask out values corresponding to encoder / decoder padding and normalise
    attention_scores_masked = torch.masked_fill(
        attention_scores_unmasked, mask=attention_mask, value=-1e8
    )

    if normalisation == "softmax":
        attention_scores = torch.softmax(attention_scores_masked, dim=1)

    elif normalisation == "sigmoid":
        attention_scores = torch.sigmoid(attention_scores_masked)

    else:
        raise ValueError(
            f"Expected `normalisation` in ['softmax', 'sigmoid'], but found {normalisation}"
        )

    contexts = torch.bmm(attention_scores.transpose(1, 2), values)

    if hard:
        hard_attention_scores = get_hard_attention_scores(
            attention_scores,
            attention_scores_masked,
            normalisation,
            deterministic_discretize,
        )
        residual_scores = torch.where(
            hard_attention_scores.bool(), attention_scores - 1, attention_scores
        )
        residual_scores = residual_scores.transpose(1, 2)
        residuals = torch.bmm(residual_scores, values)
        contexts = contexts - residuals.detach()

    else:
        hard_attention_scores = None

    return AttentionOutput(
        contexts=contexts,
        attention_scores=attention_scores,
        hard_attention_scores=hard_attention_scores,
    )
