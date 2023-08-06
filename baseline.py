import torch
import numpy as np
import torch.nn as nn

from typing import List
from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Optional
from itertools import chain
from torch.optim import AdamW
from dataclasses import dataclass
from edist.sed import standard_sed
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


Symbol = Union[str, int]
LSTM_HIDDEN = Tuple[Tensor, Tensor]


@dataclass
class Batch:
    source: Tensor
    source_length: Tensor
    target: Optional[Tensor] = None
    target_length: Optional[Tensor] = None


@dataclass
class EncoderOutput:
    encoded: Tensor


@dataclass
class DecoderOutput:
    encoded: Tensor
    new_hidden_state: LSTM_HIDDEN
    old_hidden_state: LSTM_HIDDEN


@dataclass
class AttentionOutput:
    contexts: Tensor
    attention_scores: Tensor
    attention_mask: Tensor


@dataclass
class AdditionalInferenceInformation:
    attention_scores: Tensor


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


def make_mask_2d(lengths: Tensor) -> Tensor:
    """Create binary mask from lengths indicating which indices are padding"""
    # Make sure `lengths` is a 1d array
    assert len(lengths.shape) == 1

    max_length = torch.amax(lengths, dim=0).item()
    mask = torch.arange(max_length).expand(
        (lengths.shape[0], max_length)
    )  # Shape batch x timesteps
    mask = torch.ge(mask, lengths.unsqueeze(1))

    return mask


def make_mask_3d(source_lengths: Tensor, target_lengths: Tensor) -> Tensor:
    """
    Make binary mask indicating which combinations of indices involve at least 1 padding element.
    Can be used to mask, for example, a batch attention matrix between 2 sequences
    """
    # Calculate binary masks for source and target
    # Then invert boolean values and convert to long (necessary for bmm later)
    source_mask = (~make_mask_2d(source_lengths)).long()
    target_mask = (~make_mask_2d(target_lengths)).long()

    # Add dummy dimensions for bmm
    source_mask = source_mask.unsqueeze(2)
    target_mask = target_mask.unsqueeze(1)

    # Calculate combinations by batch matrix multiplication
    mask = torch.bmm(source_mask, target_mask).bool()
    # Invert boolean values
    mask = torch.logical_not(mask)
    return mask


class BiLSTMEncoder(nn.Module):
    """
    Implements a wrapper around pytorch's LSTM for easier sequence processing.
    Note: This implementation uses trainable initialisations of hidden states, if they are not provided.
    Note: This implementation projects the combined hidden states of the forward and backward LSTMs to the common
          hidden dimension
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        projection_dim: Optional[int] = None,
    ):
        super(BiLSTMEncoder, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.projection_dim = projection_dim

        # Make properties
        self._output_size = 2 * self.hidden_size

        # Initialise modules
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        if self.projection_dim is not None:
            self.reduce_dim = nn.Linear(2 * self.hidden_size, self.projection_dim)
        else:
            self.reduce_dim = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # Initialise trainable hidden state initialisations
        self.h_0 = nn.Parameter(torch.zeros(2 * self.num_layers, 1, self.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(2 * self.num_layers, 1, self.hidden_size))

    def forward(self, inputs: Tensor, lengths: Tensor) -> EncoderOutput:
        batch_size = len(lengths)

        # Pack sequence
        lengths = torch.clamp(
            lengths, 1
        )  # Enforce all lengths are >= 1 (required by pytorch)
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )

        # Prepare hidden states
        h_0 = self.h_0.tile((1, batch_size, 1))
        c_0 = self.c_0.tile((1, batch_size, 1))

        # Apply LSTM
        encoded, _ = self.lstm(inputs, (h_0, c_0))
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        # Downsample
        encoded = self.reduce_dim(encoded)

        return EncoderOutput(encoded=encoded)


class LSTMDecoder(nn.Module):
    """
    Implements a wrapper around pytorch's LSTM (unidirectional) for easier sequence processing.
    Note: This implementation uses trainable initialisations of hidden states, if they are not provided.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(LSTMDecoder, self).__init__()

        # Save arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialise modules
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=(dropout if num_layers > 1 else 0.0),
        )

        # Initialise trainable hidden state initialisations
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
        hidden_state: Optional[LSTM_HIDDEN] = None,
    ) -> DecoderOutput:
        batch_size = len(lengths)

        # Pack sequence
        lengths = torch.clamp(
            lengths, 1
        )  # Enforce all lengths are >= 1 (required by pytorch)
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )

        # Prepare hidden states
        if hidden_state is None:
            h_0 = self.h_0.tile((1, batch_size, 1))
            c_0 = self.c_0.tile((1, batch_size, 1))
        else:
            h_0, c_0 = hidden_state

        # Apply LSTM
        encoded, new_hidden_state = self.lstm(inputs, (h_0, c_0))
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        output = DecoderOutput(
            encoded=encoded,
            new_hidden_state=new_hidden_state,
            old_hidden_state=hidden_state,
        )
        return output


class Seq2SeqModel(LightningModule):
    def __init__(
        self,
        source_alphabet_size: int,
        target_alphabet_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        scheduler_gamma: float = 1.0,
        embedding_size: int = 128,
        max_decoding_length: int = 100,
        bridge: bool = True,
    ):
        super().__init__()

        # Store Arguments
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.scheduler_gamma = scheduler_gamma
        self.max_decoding_length = max_decoding_length
        self.bridge = bridge

        self._check_arguments()
        self.save_hyperparameters()

        # Initialise Embeddings
        self.source_embeddings = nn.Embedding(
            num_embeddings=self.source_alphabet_size,
            embedding_dim=self.embedding_size,
            padding_idx=0,
        )
        self.target_embeddings = nn.Embedding(
            num_embeddings=self.target_alphabet_size,
            embedding_dim=self.embedding_size,
            padding_idx=0,
        )

        # Initialise Encoder
        self.encoder = BiLSTMEncoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            projection_dim=self.hidden_size,
        )

        # Initialise Decoder
        self.decoder = LSTMDecoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        # Initialise Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.target_alphabet_size),
        )

        # Initialise bridge mapping (optional)
        if self.bridge:
            self.bridge_mapper = nn.Linear(
                self.hidden_size, 2 * self.num_layers * self.hidden_size
            )

        # Initialise Cross Entropy Loss
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    def _check_arguments(self):
        assert (
            isinstance(self.source_alphabet_size, int) and self.source_alphabet_size > 0
        )
        assert (
            isinstance(self.target_alphabet_size, int) and self.target_alphabet_size > 0
        )
        assert isinstance(self.embedding_size, int) and self.embedding_size > 0
        assert isinstance(self.hidden_size, int) and self.hidden_size > 0
        assert isinstance(self.num_layers, int) and 0 < self.num_layers < 5
        assert isinstance(self.dropout, float) and 0.0 <= self.dropout < 1.0
        assert (
            isinstance(self.scheduler_gamma, float)
            and 0.0 < self.scheduler_gamma <= 1.0
        )
        assert (
            isinstance(self.max_decoding_length, int) and self.max_decoding_length > 0
        )
        assert isinstance(self.bridge, bool)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), weight_decay=0.0, lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def attention(
        encoder_states: Tensor,
        decoder_states: Tensor,
        encoder_lengths: Tensor,
        decoder_lengths: Tensor,
    ):
        """
        Computes dot product attention scores between encoder and decoder states and computes corresponding context
        vectors
        """
        # encoder_states: shape [batch x timesteps encoder x hidden]
        # decoder_states: shape [batch x timesteps decoder x hidden]

        # Get attention mask
        attention_mask = make_mask_3d(encoder_lengths, decoder_lengths)

        # Compute dot product of encoder states and decoder states
        decoder_states = decoder_states.transpose(1, 2)
        raw_attention_scores = torch.bmm(encoder_states, decoder_states)
        # raw_attention_scores: shape [batch x timesteps encoder x timesteps decoder]

        # Mask attention scores that correspond to padding
        attention_mask = attention_mask.to(raw_attention_scores.device)
        raw_attention_scores = torch.masked_fill(
            raw_attention_scores, mask=attention_mask, value=-1e9
        )

        # Normalise attention scores by softmax over encoder states
        attention_scores = torch.softmax(raw_attention_scores, dim=1)

        # Compute context vectors
        attention_scores = attention_scores.transpose(1, 2)
        contexts = torch.bmm(attention_scores, encoder_states)

        output = AttentionOutput(
            contexts=contexts,
            attention_scores=attention_scores,
            attention_mask=attention_mask,
        )

        return output

    def encode_source(self, source: Tensor, source_length: Tensor) -> Tensor:
        # Embed and encode source
        source_embedded = self.source_embeddings(source)
        source_encoded = self.encoder(source_embedded, source_length)
        source_encoded = source_encoded.encoded

        return source_encoded

    def encode_target(self, target: Tensor, target_length: Tensor, decoder_hidden=None):
        # Embed and Encode targets
        target_embedded = self.target_embeddings(target)
        decoder_output: DecoderOutput = self.decoder(
            target_embedded, target_length, hidden_state=decoder_hidden
        )

        return decoder_output.encoded, decoder_output.new_hidden_state

    def get_initial_decoder_hidden(
        self, source_encoded, source_length
    ) -> Tuple[Tensor, Tensor]:
        assert self.bridge

        source_mask = make_mask_2d(lengths=source_length)
        source_mask = source_mask.to(source_encoded.device)
        source_mask = source_mask.unsqueeze(-1)
        masked_source = torch.masked_fill(source_encoded, mask=source_mask, value=-1e8)
        max_pooled_source = torch.max(masked_source, dim=1).values

        initial_decoder_hidden = self.bridge_mapper(max_pooled_source)
        initial_decoder_hidden = initial_decoder_hidden.reshape(
            2, self.num_layers, -1, self.hidden_size
        )
        initial_decoder_hidden = (initial_decoder_hidden[0], initial_decoder_hidden[1])
        return initial_decoder_hidden

    @staticmethod
    def unpack_batch(batch: Batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        source = batch.source
        target = batch.target
        source_length = batch.source_length
        target_length = batch.target_length

        if source_length is not None:
            source_length = source_length.cpu()
        if target_length is not None:
            target_length = target_length.cpu()

        return source, target, source_length, target_length

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        # Unpack batch container
        source, target, source_length, target_length = self.unpack_batch(batch=batch)

        # Encode sources and targets
        source_encoded = self.encode_source(source=source, source_length=source_length)
        if self.bridge:
            initial_decoder_hidden = self.get_initial_decoder_hidden(
                source_encoded, source_length
            )
        else:
            initial_decoder_hidden = None

        target_encoded, _ = self.encode_target(
            target=target,
            target_length=target_length,
            decoder_hidden=initial_decoder_hidden,
        )

        # Compute context vectors
        attention_output = self.attention(
            source_encoded, target_encoded, source_length, target_length
        )

        # Compute prediction scores
        classifier_inputs = torch.cat(
            [target_encoded, attention_output.contexts], dim=2
        )
        prediction_scores = self.classifier(classifier_inputs)

        # Compute Loss
        labels = torch.flatten(target[:, 1:])
        scores = torch.reshape(
            prediction_scores[:, :-1, :], shape=(-1, self.target_alphabet_size)
        )
        loss = self.cross_entropy(scores, labels)

        return loss

    @staticmethod
    def compute_metrics(prediction: List[int], target: List[int]) -> Metrics:
        correct = prediction == target
        edit_distance = standard_sed(prediction, target)
        normalised_edit_distance = edit_distance / len(target)

        return Metrics(
            correct=correct,
            edit_distance=edit_distance,
            normalised_edit_distance=normalised_edit_distance,
        )

    def predict_and_evaluate(
        self,
        sources: Tensor,
        targets: Tensor,
        source_lengths: Tensor,
        target_lengths: Tensor,
    ) -> List[Metrics]:
        inference_outputs: List[InferenceOutput] = self.greedy_decode(
            source=sources, source_length=source_lengths
        )

        # Convert Targets to List
        target_lengths = target_lengths.detach().cpu().tolist()
        targets = targets.detach().cpu().tolist()
        targets = [
            target[:target_length]
            for target, target_length in zip(targets, target_lengths)
        ]

        metrics = [
            self.compute_metrics(output.prediction, target)
            for output, target in zip(inference_outputs, targets)
        ]
        return metrics

    def evaluation_step(self, batch: Batch) -> List[Metrics]:
        # Unpack batch container
        source, target, source_length, target_length = self.unpack_batch(batch=batch)

        return self.predict_and_evaluate(
            sources=source,
            targets=target,
            source_lengths=source_length,
            target_lengths=target_length,
        )

    def evaluation_epoch_end(self, eval_prefix: str, outputs: List[List[Metrics]]):
        # Flatten Metrics
        metrics = list(chain.from_iterable(outputs))

        # Aggregate Metrics
        wer = 1 - np.mean([output.correct for output in metrics]).item()
        edit_distance = np.mean([output.edit_distance for output in metrics]).item()
        normalised_edit_distance = np.mean(
            [output.normalised_edit_distance for output in metrics]
        ).item()

        self.log(f"{eval_prefix}_wer", 100 * wer)
        self.log(f"{eval_prefix}_edit_distance", edit_distance)
        self.log(f"{eval_prefix}_normalised_edit_distance", normalised_edit_distance)

    def validation_step(self, batch: Batch, batch_idx: int) -> List[Metrics]:
        return self.evaluation_step(batch=batch)

    def validation_epoch_end(self, outputs: List[List[Metrics]]) -> None:
        self.evaluation_epoch_end(eval_prefix="val", outputs=outputs)

    def test_step(self, batch: Batch, batch_idx: int) -> List[Metrics]:
        return self.evaluation_step(batch=batch)

    def test_epoch_end(self, outputs: List[List[Metrics]]) -> None:
        self.evaluation_epoch_end(eval_prefix="test", outputs=outputs)

    def predict_step(
        self, batch: Batch, batch_idx: int, dataloader_idx: Optional[int] = 0
    ) -> List[InferenceOutput]:
        return self.greedy_decode(
            source=batch.source, source_length=batch.source_length.cpu()
        )

    def greedy_decode(
        self, source: Tensor, source_length: Tensor
    ) -> List[InferenceOutput]:
        # Define constants
        batch_size = source.shape[0]
        source_timesteps = source.shape[1]
        sos_index = 2
        eos_index = 3

        # Encode Source
        source_encoded = self.encode_source(source=source, source_length=source_length)

        # Prepare Predictions
        prediction = [
            torch.full(
                (batch_size, 1),
                fill_value=sos_index,
                device=self.device,
                dtype=torch.long,
            )
        ]
        attention_scores = [
            torch.zeros(
                (batch_size, source_timesteps, 1), device=self.device, dtype=torch.long
            )
        ]
        finished = torch.zeros(batch_size, device=source_encoded.device).bool()
        prediction_length = torch.ones(
            batch_size, dtype=torch.long, device=source_encoded.device
        )
        last_prediction_length = torch.ones_like(source_length)

        if self.bridge:
            decoder_hidden = self.get_initial_decoder_hidden(
                source_encoded, source_length
            )
        else:
            decoder_hidden = None

        for t in range(self.max_decoding_length):
            # Get last prediction
            last_prediction = prediction[-1]

            # Encode last prediction
            prediction_encoded, decoder_hidden = self.encode_target(
                target=last_prediction,
                target_length=last_prediction_length,
                decoder_hidden=decoder_hidden,
            )
            # prediction_encoded: shape [batch x 1 x hidden]

            # Compute attention
            attention_output = self.attention(
                source_encoded,
                prediction_encoded,
                source_length,
                last_prediction_length,
            )
            attention_scores_t = attention_output.attention_scores
            attention_scores_t = attention_scores_t.transpose(1, 2)
            contexts = attention_output.contexts

            # Compute prediction scores
            classifier_inputs = torch.cat([prediction_encoded, contexts], dim=2)
            prediction_scores = self.classifier(classifier_inputs)

            # Save predictions and alignments
            next_predictions = torch.argmax(prediction_scores, dim=-1)
            prediction.append(next_predictions)
            attention_scores.append(attention_scores_t)

            # Update prediction length
            prediction_length = prediction_length + torch.logical_not(finished).long()

            # Stop if EOS is predicted for all sequences in batch
            eos_predicted = torch.eq(next_predictions, eos_index)
            eos_predicted = eos_predicted.flatten()
            finished = torch.logical_or(eos_predicted, finished)

            if torch.all(finished):
                break

        prediction = torch.cat(prediction, dim=1).cpu()
        prediction_length = prediction_length.detach().cpu().long().tolist()
        source_length = source_length.detach().cpu().tolist()
        attention_scores = torch.cat(attention_scores, dim=2)
        source = source.detach().cpu()

        outputs = []

        for k in range(batch_size):
            prediction_length_k = prediction_length[k]
            prediction_k = prediction[k, :prediction_length_k].tolist()
            source_k = source[k, : source_length[k]]
            attention_scores_k = attention_scores[
                k, : source_length[k], :prediction_length_k
            ]
            attention_scores_k = attention_scores_k.cpu()

            additional_information_k = AdditionalInferenceInformation(
                attention_scores=attention_scores_k
            )
            output_k = InferenceOutput(
                source=source_k,
                prediction=prediction_k,
                additional_information=additional_information_k,
            )
            outputs.append(output_k)

        return outputs
