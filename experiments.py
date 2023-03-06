import os
import json
import torch
import logging

from tqdm.auto import tqdm
from typing import Optional
from itertools import chain
from functools import partial
from torchtext.vocab import Vocab
from typing import List, Dict, Any
from data import Seq2SeqDataModule
from data import InflectionDataModule
from pytorch_lightning import Trainer
from model import InterpretableTransducer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def unbatch_predictions(results: List[Dict[str, Any]]) -> Dict[str, Optional[List[Any]]]:
    predictions = list(chain.from_iterable([batch_results["predictions"] for batch_results in results]))
    alignments = list(chain.from_iterable([batch_results["alignments"] for batch_results in results]))

    sequence_features = [batch_results["sequence_features"] for batch_results in results]
    if any(features is None for features in sequence_features):
        sequence_features = None
    else:
        sequence_features = list(chain.from_iterable(sequence_features))

    symbol_features = [batch_results["symbol_features"] for batch_results in results]
    if any(features is None for features in symbol_features):
        symbol_features = None
    else:
        symbol_features = list(chain.from_iterable(symbol_features))

    decoder_states = [batch_results["symbol_features"] for batch_results in results]
    if any(states is None for states in decoder_states):
        decoder_states = None
    else:
        decoder_states = list(chain.from_iterable(decoder_states))

    return {
        "predictions": predictions,
        "alignments": alignments,
        "sequence_features": symbol_features,
        "symbol_features": sequence_features,
        "decoder_states": decoder_states
    }


def add_sos_eos_tokens(sequences: List[List[str]]) -> List[List[str]]:
    sos_token = "[SOS]"
    eos_token = "[EOS]"
    return [[sos_token] + sequence + [eos_token] for sequence in sequences]


def decode(sequences: List[List[int]], tokenizer: Vocab) -> List[List[str]]:
    return [[tokenizer.lookup_token(symbol_id) for symbol_id in sequence] for sequence in sequences]


def get_data_modules():
    all_lang_codes = list(sorted(set([file.split(".")[0] for file in os.listdir("./data/")])))
    for lang_code in all_lang_codes:
        data_module = InflectionDataModule.from_files(
            train_path=f"data/{lang_code}.trn",
            dev_path=f"data/{lang_code}.dev",
            test_path=f"data/{lang_code}.covered.dev",
        )

        yield f"inflection-{lang_code}", data_module


def experiment(name: str, dm: Seq2SeqDataModule, binary_attention: bool, num_symbol_features: int,
               num_source_features: int):
    base_path = os.path.join("./results/", name)
    check_val_every_n_epoch = 1

    logger = pl_loggers.CSVLogger(save_dir=os.path.join(base_path, "logs"))
    early_stopping_callback = EarlyStopping(monitor="val_edit_distance", patience=3, mode="min", verbose=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_path, "saved_models"),
        filename=name + "-{val_edit_distance}",
        monitor="val_edit_distance",
        save_last=True,
        save_top_k=1,
        mode="min",
        verbose=False
    )

    dm.prepare_data()
    dm.setup(stage="fit")

    model = InterpretableTransducer(
        source_alphabet_size=dm.source_alphabet_size, target_alphabet_size=dm.target_alphabet_size,
        num_layers=2, dropout=0.0, hidden_size=256, max_decoding_length=100,
        num_source_features=num_source_features, num_symbol_features=num_symbol_features, num_decoder_states=0,
        enable_seq2seq_loss=True, binary_attention=binary_attention
    )

    trainer = Trainer(
        max_epochs=500, log_every_n_steps=10, check_val_every_n_epoch=check_val_every_n_epoch, accelerator='gpu',
        devices=1, gradient_clip_val=1.0, enable_progress_bar=False, logger=logger, enable_model_summary=False,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    model.load_from_checkpoint(checkpoint_path=os.path.join(base_path, "saved_models", "last.ckpt"))

    dm.setup(stage="test")

    # Get Predictions
    train_predictions = unbatch_predictions(
        trainer.predict(model=model, dataloaders=dm.train_dataloader(shuffle=False))
    )
    validation_predictions = unbatch_predictions(trainer.predict(model=model, dataloaders=dm.val_dataloader()))
    test_predictions = unbatch_predictions(trainer.predict(model=model, dataloaders=dm.test_dataloader()))

    # Decode Predictions
    decode_target = partial(decode, tokenizer=dm.target_tokenizer)
    train_predictions["predictions"] = decode_target(train_predictions["predictions"])
    validation_predictions["predictions"] = decode_target(validation_predictions["predictions"])
    test_predictions["predictions"] = decode_target(test_predictions["predictions"])

    # Get Sources and Targets
    train_sources, train_targets = zip(*dm.train_data)
    validation_sources, validation_targets = zip(*dm.dev_data)
    test_sources, test_targets = zip(*dm.test_data)

    train_sources = add_sos_eos_tokens(train_sources)
    train_targets = add_sos_eos_tokens(train_targets)
    validation_sources = add_sos_eos_tokens(validation_sources)
    validation_targets = add_sos_eos_tokens(validation_targets)
    test_sources = add_sos_eos_tokens(test_sources)
    test_targets = add_sos_eos_tokens(test_targets)

    # Add Sources & Targets to predictions dicts
    train_predictions["sources"] = train_sources
    train_predictions["targets"] = train_targets
    validation_predictions["sources"] = validation_sources
    validation_predictions["targets"] = validation_targets
    test_predictions["sources"] = test_sources
    test_predictions["targets"] = test_targets

    # Save
    predictions = {"train": train_predictions, "validation": validation_predictions, "test": test_predictions}
    with open(os.path.join(base_path, "predictions.json"), "w") as sf:
        json.dump(predictions, sf)


if __name__ == '__main__':
    # Global Settings
    torch.set_float32_matmul_precision('medium')
    logging.disable(logging.WARNING)

    # Progress Bar
    progress_bar = tqdm(desc="Progress", total=(len(os.listdir("./data/")) // 3))
    progress_bar.display()

    for task_name, task_data in get_data_modules():
        for use_binary_attention in [True]:
            for n_symbol_features, n_source_features in [(0, 0)]:
                for trial in range(1, 2):
                    experiment_name = f"{task_name}-binary_attention={use_binary_attention}"
                    experiment_name += f"-num_symbol_features={n_symbol_features}"
                    experiment_name += f"-num_source_features={n_source_features}"
                    experiment_name += f"-trial={trial}"

                    experiment(
                        name=experiment_name, dm=task_data, binary_attention=use_binary_attention,
                        num_symbol_features=n_symbol_features, num_source_features=n_source_features
                    )
                    progress_bar.update(1)

    progress_bar.close()
