from collections import namedtuple

EncoderOutput = namedtuple(
    "EncoderOutput", field_names=["source_embeddings", "source_encodings"]
)
DecoderOutput = namedtuple(
    "DecoderOutput", field_names=[
        "contexts", "seq2seq_contexts", "hidden_state", "source_selection", "decoder_outputs", "decoder_states",
        "decoder_state_selection", "target_embedded"
    ]
)
BridgeOutput = namedtuple("BridgeOutput", field_names=["output", "feature_scores"])
AttentionOutput = namedtuple("AttentionOutput", field_names=["contexts", "attention_scores", "hard_attention_scores"])
MaskContainer = namedtuple(
    "MaskContainer", field_names=["source_mask", "target_mask", "attention_mask"]
)
InferenceOutput = namedtuple(
    "InferenceOutput", field_names=[
        "predictions", "alignments", "sequence_features", "symbol_features", "decoder_states"
    ]
)
MetricsContainer = namedtuple("MetricsContainer", field_names=["correct", "edit_distance", "normalised_edit_distance"])
ValidationContainer = namedtuple("ValidationContainer", field_names=["metrics", "predictions", "targets", "sources"])
