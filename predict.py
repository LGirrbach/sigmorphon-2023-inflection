from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from itertools import chain
from functools import partial
from dataclasses import asdict
from torchtext.vocab import Vocab
from pytorch_lightning import Trainer
from data import InflectionDataModule
from containers import InferenceOutput
from model import InterpretableTransducer


def _un_batch_predictions(
    results: List[List[InferenceOutput]],
) -> Dict[str, Optional[List[Any]]]:
    # Flatten Results
    results = list(chain.from_iterable(results))

    sources = [sample_result.source for sample_result in results]
    predictions = [sample_result.prediction for sample_result in results]
    additional_information = [
        asdict(sample_result.additional_information) for sample_result in results
    ]

    return {
        "sources": sources,
        "predictions": predictions,
        "additional_information": additional_information,
    }


def _add_sos_eos_tokens(sequences: List[List[str]]) -> List[List[str]]:
    sos_token = "[SOS]"
    eos_token = "[EOS]"
    return [[sos_token] + sequence + [eos_token] for sequence in sequences]


def _decode(sequences: List[List[int]], tokenizer: Vocab) -> List[List[str]]:
    return [
        [tokenizer.lookup_token(symbol_id) for symbol_id in sequence]
        for sequence in sequences
    ]


def predict(
    trainer: Trainer, model: InterpretableTransducer, dataset: InflectionDataModule
):
    dataset.setup(stage="test")

    # Get Predictions
    # train_predictions = trainer.predict(model=model, dataloaders=dataset.train_dataloader(shuffle=False))
    # train_predictions = _un_batch_predictions(train_predictions)
    validation_predictions = trainer.predict(
        model=model, dataloaders=dataset.val_dataloader()
    )
    validation_predictions = _un_batch_predictions(validation_predictions)
    test_predictions = trainer.predict(
        model=model, dataloaders=dataset.test_dataloader()
    )
    test_predictions = _un_batch_predictions(test_predictions)

    # Decode Predictions
    decode_target = partial(_decode, tokenizer=dataset.target_tokenizer)
    # train_predictions["predictions"] = decode_target(train_predictions["predictions"])
    validation_predictions["predictions"] = decode_target(
        validation_predictions["predictions"]
    )
    test_predictions["predictions"] = decode_target(test_predictions["predictions"])

    # Get Sources and Targets
    # train_sources, train_targets = zip(*dataset.train_data)
    validation_sources, validation_targets = zip(*dataset.dev_data)
    test_sources, test_targets = zip(*dataset.test_data)

    # train_sources = _add_sos_eos_tokens(train_sources)
    # train_targets = _add_sos_eos_tokens(train_targets)
    validation_sources = _add_sos_eos_tokens(validation_sources)
    validation_targets = _add_sos_eos_tokens(validation_targets)
    test_sources = _add_sos_eos_tokens(test_sources)
    test_targets = _add_sos_eos_tokens(test_targets)

    # Add Sources & Targets to predictions dicts
    # train_predictions["sources"] = train_sources
    # train_predictions["targets"] = train_targets
    validation_predictions["sources"] = validation_sources
    validation_predictions["targets"] = validation_targets
    test_predictions["sources"] = test_sources
    test_predictions["targets"] = test_targets

    # Save
    # predictions = {"train": train_predictions, "validation": validation_predictions, "test": test_predictions}
    predictions = {"validation": validation_predictions, "test": test_predictions}
    return predictions
