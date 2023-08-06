from typing import List
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    batch_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    scheduler_gamma: float


@dataclass
class Batch:
    source: Tensor
    source_length: Tensor
    target: Optional[Tensor] = None
    target_length: Optional[Tensor] = None


@dataclass
class EncoderOutput:
    source_embeddings: Tensor
    source_encodings: Tensor


@dataclass
class DecoderOutput:
    contexts: Tensor
    seq2seq_contexts: Tensor
    hidden_state: Tensor
    source_selection: Tensor
    decoder_outputs: Tensor
    decoder_states: Tensor
    decoder_state_selection: Tensor
    target_embedded: Tensor


@dataclass
class BridgeOutput:
    output: Tensor
    feature_scores: Tensor


@dataclass
class AttentionOutput:
    contexts: Tensor
    attention_scores: Tensor
    hard_attention_scores: Tensor


@dataclass
class MaskContainer:
    source_mask: Tensor
    target_mask: Tensor
    attention_mask: Tensor


@dataclass
class AdditionalInferenceInformation:
    alignment: Tensor
    sequence_features: Tensor
    symbol_features: Tensor
    decoder_states: Tensor


@dataclass
class InferenceOutput:
    source: List[int]
    prediction: List[int]
    additional_information: AdditionalInferenceInformation


@dataclass
class Metrics:
    correct: bool
    edit_distance: float
    normalised_edit_distance: float
