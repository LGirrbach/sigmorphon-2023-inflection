import torch
import numpy as np
import torch.nn as nn

from typing import List
from typing import Tuple
from torch import Tensor
from typing import Optional
from itertools import chain
from torch.optim import AdamW
from decoder import LSTMDecoder
from attention import attention
from bridge import EncoderBridge
from encoder import BiLSTMEncoder
from edist.sed import standard_sed
from containers import MaskContainer
from containers import EncoderOutput
from containers import DecoderOutput
from utils import discretize_sigmoid
from utils import discretize_softmax
from containers import InferenceOutput
from containers import MetricsContainer
from containers import ValidationContainer
from utils import make_mask_2d, make_mask_3d
from pytorch_lightning import LightningModule


class InterpretableTransducer(LightningModule):
    def __init__(self, source_alphabet_size: int, target_alphabet_size: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.0, embedding_size: int = 128, num_source_features: int = 0,
                 num_symbol_features: int = 0, num_decoder_states: int = 0, autoregressive_order: int = 0,
                 max_decoding_length: int = 100, binary_attention: bool = True, enable_seq2seq_loss: bool = False):
        super().__init__()

        # Store Arguments
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_source_features = num_source_features
        self.num_symbol_features = num_symbol_features
        self.num_decoder_states = num_decoder_states
        self.autoregressive_order = autoregressive_order
        self.max_decoding_length = max_decoding_length
        self.binary_attention = binary_attention
        self.enable_seq2seq_loss = enable_seq2seq_loss

        self._check_arguments()
        self.save_hyperparameters()

        # Initialise Embeddings
        self.source_embeddings = nn.Embedding(
            num_embeddings=self.source_alphabet_size, embedding_dim=self.embedding_size, padding_idx=0
        )
        self.target_embeddings = nn.Embedding(
            num_embeddings=self.target_alphabet_size, embedding_dim=self.embedding_size, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # Initialise Encoder
        self.encoder = BiLSTMEncoder(
            input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, projection_dim=self.hidden_size
        )

        if self.num_symbol_features > 0:
            self.symbol_feature_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.num_symbol_features),
                nn.Sigmoid()
            )
            self.symbol_features = nn.Parameter(torch.empty(self.num_symbol_features, self.hidden_size))
            torch.nn.init.xavier_normal_(self.symbol_features)

        # Initialise Encoder -> Decoder Bridge (optional)
        self.bridge = EncoderBridge(
            hidden_size=self.hidden_size, num_source_features=self.num_source_features,
            num_decoder_layers=self.num_layers
        )
        if self.num_source_features > 0:
            self.bridge_features = nn.Parameter(torch.empty(self.num_source_features, self.hidden_size))
            torch.nn.init.xavier_normal_(self.bridge_features)

        # Initialise Decoder
        self.decoder = LSTMDecoder(
            input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout
        )
        if self.num_decoder_states > 0:
            self.decoder_state_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.num_decoder_states)
            )
            self.decoder_states = nn.Parameter(torch.empty(self.num_decoder_states, self.hidden_size))
            torch.nn.init.xavier_normal_(self.decoder_states)

        # Initialise Final Predictor
        classifier_in_size = self.embedding_size
        classifier_in_size += (self.hidden_size if self.num_symbol_features > 0 else 0)
        classifier_in_size += (self.hidden_size if self.num_source_features > 0 else 0)
        classifier_in_size += (self.hidden_size if self.num_decoder_states > 0 else 0)
        classifier_in_size += self.autoregressive_order * self.embedding_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(classifier_in_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.target_alphabet_size)
        )

        # Initialise Seq2Seq Classifier
        if self.enable_seq2seq_loss:
            self.seq2seq_classifier = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, self.target_alphabet_size)
            )

        if self.autoregressive_order > 0:
            conv_filter = []
            embedding_dim_indexer = torch.arange(self.embedding_size)
            for order in range(self.autoregressive_order):
                order_filter = torch.zeros(self.embedding_size, self.embedding_size, self.autoregressive_order)
                order_filter[embedding_dim_indexer, embedding_dim_indexer, order] = 1.
                conv_filter.append(order_filter)

            conv_filter = torch.cat(conv_filter, dim=0)
            self.register_buffer("target_embedding_fold", conv_filter)
            self.register_buffer(
                "target_embedding_padding", torch.zeros(1, self.embedding_size, self.autoregressive_order-1)
            )

        # Initialise Loss
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    def _check_arguments(self):
        assert isinstance(self.source_alphabet_size, int) and self.source_alphabet_size > 0
        assert isinstance(self.target_alphabet_size, int) and self.target_alphabet_size > 0
        assert isinstance(self.embedding_size, int) and self.embedding_size > 0
        assert isinstance(self.hidden_size, int) and self.hidden_size > 0
        assert isinstance(self.num_layers, int) and 0 < self.num_layers < 5
        assert isinstance(self.dropout, float) and 0.0 <= self.dropout < 1.0
        assert isinstance(self.num_source_features, int) and self.num_source_features >= 0
        assert isinstance(self.num_symbol_features, int) and self.num_symbol_features >= 0
        assert isinstance(self.num_decoder_states, int) and self.num_decoder_states >= 0
        assert isinstance(self.autoregressive_order, int) and self.autoregressive_order >= 0
        assert isinstance(self.max_decoding_length, int) and self.max_decoding_length > 0
        assert isinstance(self.binary_attention, bool)
        assert isinstance(self.enable_seq2seq_loss, bool)

    def encode(self, source: Tensor, source_length: Tensor) -> EncoderOutput:
        # Embed and encode source
        source_embedded = self.source_embeddings(source)
        source_embedded_with_dropout = self.embedding_dropout(source_embedded)
        source_encoded = self.encoder(source_embedded_with_dropout, source_length)

        return EncoderOutput(source_embeddings=source_embedded, source_encodings=source_encoded)

    def decode(self, source_encoded: Tensor, source_embedded: Tensor, target: Tensor, target_length: Tensor,
               attention_mask: Tensor, decoder_hidden, deterministic_discretize: bool):
        # Embed and Encode targets
        target_embedded = self.target_embeddings(target)
        target_embedded = self.embedding_dropout(target_embedded)
        decoder_output = self.decoder(target_embedded, target_length, hidden_state=decoder_hidden)
        target_encoded = decoder_output["encoded"]
        new_decoder_hidden = decoder_output["new_hidden_state"]

        # Compute attention
        encoder_attention_normalisation = "sigmoid" if self.binary_attention else "softmax"
        encoder_attention_output = attention(
            encoder_states=source_encoded, decoder_states=target_encoded, attention_mask=attention_mask,
            values=source_embedded, normalisation=encoder_attention_normalisation, hard=True,
            deterministic_discretize=deterministic_discretize
        )
        contexts = encoder_attention_output.contexts
        # contexts: shape [batch x timesteps decoder x embedding size]
        source_selection = encoder_attention_output.hard_attention_scores

        # Compute Decoder States (optional)
        if self.num_decoder_states > 0:
            decoder_state_features, hard_decoder_state_scores = self.get_decoder_states(
                target_encoded, deterministic=deterministic_discretize
            )
        else:
            decoder_state_features = None
            hard_decoder_state_scores = None

        return DecoderOutput(
            contexts=contexts, hidden_state=new_decoder_hidden, source_selection=source_selection,
            decoder_outputs=target_encoded, decoder_states=decoder_state_features,
            decoder_state_selection=hard_decoder_state_scores, target_embedded=target_embedded
        )

    def get_autoregressive_embeddings(self, target_embedded: Tensor) -> Tensor:
        assert self.autoregressive_order > 0
        batch_size, decoder_timesteps, _ = target_embedded.shape

        target_embedding_ngrams = target_embedded.transpose(1, 2)
        target_embedding_padding = self.target_embedding_padding.expand(
            batch_size, self.embedding_size, self.autoregressive_order-1
        )
        target_embedding_ngrams = torch.cat([target_embedding_padding, target_embedding_ngrams], dim=2)
        target_embedding_ngrams = nn.functional.conv1d(target_embedding_ngrams, self.target_embedding_fold)
        target_embedding_ngrams = target_embedding_ngrams.transpose(1, 2)
        return target_embedding_ngrams

    def get_prediction_scores(self, source_contexts: Tensor, source_features: Optional[Tensor] = None,
                              decoder_states: Optional[Tensor] = None) -> Tensor:
        # source_contexts: shape [batch x timesteps x embedding size]
        # source_features: shape [batch x hidden]
        #
        # Get constants
        batch_size, timesteps, _ = source_contexts.shape

        # Initialise source features
        if source_features is None:
            classifier_inputs = source_contexts
        else:
            source_features = source_features.unsqueeze(1)
            source_features = source_features.expand(batch_size, timesteps, self.hidden_size)
            classifier_inputs = torch.cat([source_contexts, source_features], dim=-1)

        if decoder_states is not None:
            classifier_inputs = torch.cat([classifier_inputs, decoder_states], dim=-1)

        # Compute classification scores
        return self.classifier(classifier_inputs)

    def compute_bridge(self, source_encodings: Tensor, source_mask: Tensor, deterministic: bool = True):
        bridge_output = self.bridge(sequences=source_encodings, mask=source_mask)
        feature_scores = bridge_output.feature_scores

        # Reformat Bridge Output
        bridge_output = bridge_output.output
        bridge_output = bridge_output.reshape(2, self.num_layers, bridge_output.shape[0], self.hidden_size)
        bridge_output = (bridge_output[0], bridge_output[1])

        if feature_scores is None:
            return bridge_output, None, None

        # Discretize Feature Scores
        hard_feature_scores = discretize_sigmoid(feature_scores, deterministic=deterministic)
        residual_scores = torch.where(hard_feature_scores.bool(), feature_scores - 1, feature_scores)

        # Compute Features
        bridge_features = torch.mm(feature_scores, self.bridge_features)
        residual_features = torch.mm(residual_scores, self.bridge_features)
        bridge_features = bridge_features - residual_features.detach()

        return bridge_output, bridge_features, hard_feature_scores

    def get_symbol_features(self, source_encoded: Tensor, deterministic: bool = True) -> Tuple[Tensor, Tensor]:
        assert self.num_symbol_features > 0
        symbol_feature_scores = self.symbol_feature_classifier(source_encoded)
        hard_symbol_feature_scores = discretize_sigmoid(symbol_feature_scores, deterministic=deterministic)
        residual_scores = torch.where(
            hard_symbol_feature_scores.bool(), symbol_feature_scores - 1, symbol_feature_scores
        )

        all_symbol_features = self.symbol_features.expand(
            source_encoded.shape[0], self.num_symbol_features, self.hidden_size
        )
        soft_symbol_features = torch.bmm(symbol_feature_scores, all_symbol_features)
        residual_symbol_features = torch.bmm(residual_scores, all_symbol_features)
        symbol_features = soft_symbol_features - residual_symbol_features.detach()

        return symbol_features, hard_symbol_feature_scores

    def get_decoder_states(self, target_encoded: Tensor, deterministic: bool = True) -> Tuple[Tensor, Tensor]:
        assert self.num_decoder_states > 0
        decoder_state_scores = self.decoder_state_classifier(target_encoded)
        hard_decoder_state_scores = discretize_softmax(decoder_state_scores, deterministic=deterministic, dim=-1)
        decoder_state_scores = torch.softmax(decoder_state_scores, dim=-1)
        residual_scores = torch.where(
            hard_decoder_state_scores.bool(), decoder_state_scores - 1, decoder_state_scores
        )

        all_decoder_state_features = self.decoder_states.expand(
            target_encoded.shape[0], self.num_decoder_states, self.hidden_size
        )
        soft_decoder_state_features = torch.bmm(decoder_state_scores, all_decoder_state_features)
        residual_decoder_state_features = torch.bmm(residual_scores, all_decoder_state_features)
        decoder_state_features = soft_decoder_state_features - residual_decoder_state_features

        return decoder_state_features, hard_decoder_state_scores

    def get_masks(self, source_lengths: Tensor, target_lengths: Tensor):
        source_mask = make_mask_2d(lengths=source_lengths).to(self.device)
        target_mask = make_mask_2d(lengths=target_lengths).to(self.device)
        attention_mask = make_mask_3d(source_lengths=source_lengths, target_lengths=target_lengths).to(self.device)

        return MaskContainer(
            source_mask=source_mask, target_mask=target_mask, attention_mask=attention_mask
        )

    def greedy_decode(self, source: Tensor, source_length: Tensor):
        # Define constants
        batch_size = source.shape[0]
        source_timesteps = source.shape[1]
        sos_index = 2
        eos_index = 3

        # Make Masks
        source_length = source_length.cpu()
        target_lengths = torch.ones_like(source_length)
        masks = self.get_masks(source_lengths=source_length, target_lengths=target_lengths)

        # Embed and encode source
        encoder_output = self.encode(source=source, source_length=source_length)
        source_embeddings = encoder_output.source_embeddings
        source_encodings = encoder_output.source_encodings

        if self.num_symbol_features > 0:
            symbol_features, hard_symbol_feature_scores = self.get_symbol_features(
                source_encodings, deterministic=True
            )
            source_embeddings = torch.cat([source_embeddings, symbol_features], dim=-1)
            hard_symbol_feature_scores = hard_symbol_feature_scores.detach().cpu().long()
        else:
            hard_symbol_feature_scores = None

        # Apply Encoder -> Decoder Bridge (optional)
        bridge_output, bridge_features, hard_bridge_features = self.compute_bridge(
            source_encodings, masks.source_mask, deterministic=False
        )

        predictions = [torch.full((batch_size, 1), fill_value=sos_index, device=self.device, dtype=torch.long)]
        alignments = [torch.zeros((batch_size, source_timesteps), device=self.device, dtype=torch.long)]
        decoder_states = [torch.full((batch_size,), fill_value=-1, device=self.device, dtype=torch.long)]
        decoder_hidden = bridge_output

        for t in range(self.max_decoding_length):
            last_prediction = predictions[-1]

            # Apply Decoder
            decoder_output = self.decode(
                source_encodings, source_embeddings, last_prediction, target_lengths,
                attention_mask=masks.attention_mask, decoder_hidden=decoder_hidden, deterministic_discretize=True
            )

            if self.autoregressive_order > 0:
                window_start = max(0, len(predictions) - self.autoregressive_order)
                autoregressive_embeddings = torch.cat(predictions[window_start:], dim=1)
                autoregressive_embeddings = self.target_embeddings(autoregressive_embeddings)
                autoregressive_embeddings = self.get_autoregressive_embeddings(autoregressive_embeddings)
                autoregressive_embeddings = autoregressive_embeddings[:, -1:, :]
                contexts = torch.cat([decoder_output.contexts, autoregressive_embeddings], dim=2)
            else:
                contexts = decoder_output.contexts

            # Get Prediction Scores
            prediction_scores = self.get_prediction_scores(
                source_contexts=contexts, source_features=bridge_features,
                decoder_states=decoder_output.decoder_states
            )

            # Get Predictions
            prediction = torch.argmax(prediction_scores, dim=-1)
            predictions.append(prediction)
            decoder_hidden = decoder_output.hidden_state

            alignments.append(decoder_output.source_selection.reshape(batch_size, source_timesteps))
            if self.num_decoder_states > 0:
                decoder_states_t = decoder_output.decoder_state_selection.detach()
                decoder_states_t = decoder_states_t.reshape(batch_size, self.num_decoder_states)
                decoder_states_t = torch.argmax(decoder_states_t, dim=-1).long()
                decoder_states.append(decoder_states_t)

        predictions = torch.cat(predictions, dim=1).detach().cpu().tolist()
        alignments = torch.stack(alignments).permute([1, 2, 0]).detach().cpu().long()
        source_length = source_length.detach().cpu().tolist()

        if self.num_decoder_states > 0:
            decoder_states = torch.stack(decoder_states).transpose(0, 1).detach().cpu().long()
        else:
            decoder_states = None

        if hard_bridge_features is not None:
            sequence_features = hard_bridge_features.detach().cpu().long().tolist()
        else:
            sequence_features = None

        truncated_predictions = []
        truncated_alignments = []
        truncated_symbol_features = []
        truncated_decoder_states = []

        for k, (prediction, source_length_k) in enumerate(zip(predictions, source_length)):
            if eos_index not in prediction:
                prediction_length = len(prediction)
            else:
                prediction_length = prediction.index(eos_index) + 1

            truncated_predictions.append(prediction[:prediction_length])
            truncated_alignments.append(alignments[k, :source_length_k, :prediction_length].tolist())

            if hard_symbol_feature_scores is not None:
                truncated_symbol_features.append(hard_symbol_feature_scores[k, :source_length_k, :].tolist())

            if decoder_states is not None:
                decoder_states_k = decoder_states[k, :prediction_length].tolist()
                truncated_decoder_states.append(decoder_states_k)

        if not truncated_symbol_features:
            truncated_symbol_features = None
        if not truncated_decoder_states:
            truncated_decoder_states = None

        return InferenceOutput(
            predictions=truncated_predictions, alignments=truncated_alignments, sequence_features=sequence_features,
            symbol_features=truncated_symbol_features, decoder_states=truncated_decoder_states
        )

    def get_seq2seq_loss(self, source_encoded: Tensor, target_encoded: Tensor, attention_mask: Tensor,
                         target: Tensor) -> Tensor:
        encoder_attention_normalisation = "sigmoid" if self.binary_attention else "softmax"
        encoder_attention_output = attention(
            encoder_states=source_encoded, decoder_states=target_encoded, attention_mask=attention_mask,
            values=source_encoded, normalisation=encoder_attention_normalisation, hard=False,
            deterministic_discretize=True
        )
        contexts = encoder_attention_output.contexts

        classifier_inputs = torch.cat([target_encoded, contexts], dim=-1)
        prediction_scores = self.seq2seq_classifier(classifier_inputs)

        # Compute Loss
        prediction_scores = prediction_scores[:, :-1, :]
        prediction_scores = prediction_scores.reshape(-1, self.target_alphabet_size)
        labels = target[:, 1:].reshape(-1)

        loss = self.cross_entropy(prediction_scores, labels)
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        torch.set_grad_enabled(True)

        source = batch["source"]
        target = batch["target"]
        source_length = batch["source_length"].cpu()
        target_length = batch["target_length"].cpu()

        # Make Masks
        masks = self.get_masks(source_lengths=source_length, target_lengths=target_length)

        # Embed and encode source
        encoder_output = self.encode(source=source, source_length=source_length)
        source_embeddings = encoder_output.source_embeddings
        source_encodings = encoder_output.source_encodings

        if self.num_symbol_features > 0:
            symbol_features, hard_symbol_feature_scores = self.get_symbol_features(
                source_encodings, deterministic=False
            )
            source_embeddings = torch.cat([source_embeddings, symbol_features], dim=-1)

        # Apply Encoder -> Decoder Bridge (optional)
        bridge_output, bridge_features, _ = self.compute_bridge(
            source_encodings, masks.source_mask, deterministic=False
        )

        # Apply Decoder
        decoder_output = self.decode(
            source_encodings, source_embeddings, target, target_length, attention_mask=masks.attention_mask,
            decoder_hidden=bridge_output, deterministic_discretize=False
        )

        if self.autoregressive_order > 0:
            target_embedding_ngrams = self.get_autoregressive_embeddings(decoder_output.target_embedded)
            # target_embedding_ngrams = target_embedding_ngrams[:, 1:, :]
            contexts = torch.cat([decoder_output.contexts, target_embedding_ngrams], dim=2)
        else:
            contexts = decoder_output.contexts

        # Get Prediction Scores
        prediction_scores = self.get_prediction_scores(
            source_contexts=contexts, source_features=bridge_features, decoder_states=decoder_output.decoder_states
        )

        # Compute Loss
        prediction_scores = prediction_scores[:, :-1, :]
        prediction_scores = prediction_scores.reshape(-1, self.target_alphabet_size)
        labels = target[:, 1:].reshape(-1)

        loss = self.cross_entropy(prediction_scores, labels)

        if self.enable_seq2seq_loss:
            seq2seq_loss = self.get_seq2seq_loss(
                source_encodings, decoder_output.decoder_outputs, masks.attention_mask, target
            )
            loss = loss + seq2seq_loss - seq2seq_loss.detach()

        return loss

    @staticmethod
    def compute_metrics(prediction: List[int], target: List[int]):
        correct = prediction == target
        edit_distance = standard_sed(prediction, target)

        return MetricsContainer(correct=correct, edit_distance=edit_distance)

    def predict_and_evaluate(self, sources: Tensor, targets: Tensor, source_lengths: Tensor,
                             target_lengths: Tensor) -> ValidationContainer:
        inference_output = self.greedy_decode(source=sources, source_length=source_lengths)

        # Prepare Sources & Targets
        source_lengths = source_lengths.detach().cpu().tolist()
        sources = sources.detach().cpu().tolist()
        sources = [source[:source_length] for source, source_length in zip(sources, source_lengths)]

        target_lengths = target_lengths.detach().cpu().tolist()
        targets = targets.detach().cpu().tolist()
        targets = [target[:target_length] for target, target_length in zip(targets, target_lengths)]

        metrics = []

        for prediction, target in zip(inference_output.predictions, targets):
            instance_metrics = self.compute_metrics(prediction, target)
            metrics.append(instance_metrics)

        return ValidationContainer(metrics=metrics, predictions=inference_output, targets=targets, sources=sources)

    def evaluation_step(self, batch) -> ValidationContainer:
        return self.predict_and_evaluate(
            sources=batch["source"], targets=batch["target"], source_lengths=batch["source_length"],
            target_lengths=batch["target_length"]
        )

    def validation_step(self, batch, batch_idx) -> ValidationContainer:
        self.eval()
        torch.set_grad_enabled(False)

        return self.evaluation_step(batch=batch)

    def validation_epoch_end(self, outputs: List[ValidationContainer]) -> None:
        metrics = [output.metrics for output in outputs]
        metrics = list(chain.from_iterable(metrics))
        wer = 1 - np.mean([output.correct for output in metrics]).item()
        edit_distance = np.mean([output.edit_distance for output in metrics]).item()

        self.log("val_wer", 100 * wer)
        self.log("val_edit_distance", edit_distance)

    def test_step(self, batch, batch_idx) -> ValidationContainer:
        self.eval()
        torch.set_grad_enabled(False)

        return self.evaluation_step(batch=batch)

    def test_epoch_end(self, outputs: List[ValidationContainer]) -> None:
        # Compute Test Metrics
        metrics = [output.metrics for output in outputs]
        metrics = list(chain.from_iterable(metrics))
        wer = 1 - np.mean([output.correct for output in metrics]).item()
        edit_distance = np.mean([output.edit_distance for output in metrics]).item()

        self.log("test_wer", 100 * wer)
        self.log("test_edit_distance", edit_distance)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = 0):
        self.eval()
        torch.set_grad_enabled(False)

        inference_output = self.greedy_decode(source=batch["source"], source_length=batch["source_length"])
        return inference_output._asdict()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), weight_decay=0.0, lr=0.001)
        return optimizer
