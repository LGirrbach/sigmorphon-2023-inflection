import os
import json
import torch
import shutil
import logging
import argparse
import numpy as np
import pandas as pd

from predict import predict
from data import InflectionDataModule
from pytorch_lightning import Trainer
from containers import Hyperparameters
from model import InterpretableTransducer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def _load_dataset(language: str, data_path: str) -> InflectionDataModule:
    data_module = InflectionDataModule.from_files(
        train_path=os.path.join(data_path, f"{language}.trn"),
        dev_path=os.path.join(data_path, f"{language}.dev"),
        test_path=os.path.join(data_path, f"{language}.covered.dev"),
    )
    return data_module


def _represent_hyperparameter_value(value):
    if isinstance(value, float):
        return np.round(value, 2).item()
    else:
        return value


def _make_experiment_name(language: str, num_symbol_features: int, num_source_features: int,
                          autoregressive_order: int, hyperparameters: Hyperparameters) -> str:
    experiment_name = language
    experiment_name = experiment_name + "-" + f"num_symbol_features={num_symbol_features}"
    experiment_name = experiment_name + "-" + f"num_source_features={num_source_features}"
    experiment_name = experiment_name + "-" + f"autoregressive_order={autoregressive_order}"

    hyperparameter_string = [
        (param, _represent_hyperparameter_value(value)) for param, value in hyperparameters._asdict().items()
    ]
    hyperparameter_string = [f"{param}={value}" for param, value in hyperparameter_string]
    hyperparameter_string = "-".join(hyperparameter_string)
    experiment_name = experiment_name + "-" + hyperparameter_string
    return experiment_name


def _check_arguments(num_symbol_features: int, num_source_features: int, autoregressive_order: int,
                     hyperparameters: Hyperparameters) -> None:
    assert isinstance(num_symbol_features, int) and num_symbol_features >= 0
    assert isinstance(num_source_features, int) and num_source_features >= 0
    assert isinstance(autoregressive_order, int) and autoregressive_order >= 0

    assert isinstance(hyperparameters.batch_size, int) and hyperparameters.batch_size >= 1
    assert isinstance(hyperparameters.num_layers, int) and hyperparameters.num_layers >= 1
    assert isinstance(hyperparameters.hidden_size, int) and hyperparameters.hidden_size >= 1
    assert isinstance(hyperparameters.dropout, float) and 0.0 <= hyperparameters.dropout <= 1.0
    assert isinstance(hyperparameters.scheduler_gamma, float) and hyperparameters.scheduler_gamma > 0.0


def _make_callbacks(base_path: str, experiment_name: str):
    early_stopping_callback = EarlyStopping(
        monitor="val_normalised_edit_distance", patience=3, mode="min", verbose=False
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_path, "saved_models"),
        filename=experiment_name + "-{val_normalised_edit_distance}",
        monitor="val_normalised_edit_distance",
        save_last=True,
        save_top_k=1,
        mode="min",
        verbose=False
    )

    return early_stopping_callback, checkpoint_callback


def _make_model(dataset: InflectionDataModule, hyperparameters: Hyperparameters, num_symbol_features: int,
                num_source_features: int, autoregressive_order: int) -> InterpretableTransducer:
    return InterpretableTransducer(
        source_alphabet_size=dataset.source_alphabet_size,
        target_alphabet_size=dataset.target_alphabet_size,
        num_layers=hyperparameters.num_layers,
        hidden_size=hyperparameters.hidden_size,
        dropout=hyperparameters.dropout,
        scheduler_gamma=hyperparameters.scheduler_gamma,
        num_source_features=num_source_features,
        num_symbol_features=num_symbol_features,
        autoregressive_order=autoregressive_order,
        enable_seq2seq_loss=True,
    )


def experiment(base_path: str, data_path: str, language: str, num_symbol_features: int, num_source_features: int,
               autoregressive_order: int, hyperparameters: Hyperparameters, overwrite: bool = False,
               get_predictions: bool = True, verbose: bool = False):
    # Global Settings
    torch.set_float32_matmul_precision('medium')

    if not verbose:
        logging.disable(logging.WARNING)

    # Check Arguments
    _check_arguments(num_symbol_features, num_source_features, autoregressive_order, hyperparameters)

    # Make Experiment Name and Base Path
    experiment_name = _make_experiment_name(
        language, num_symbol_features, num_source_features, autoregressive_order, hyperparameters
    )
    base_path = os.path.join(base_path, experiment_name)

    if os.path.exists(base_path) and not overwrite:
        raise FileExistsError(f"Model Path {base_path} exists.")
    elif os.path.exists(base_path) and overwrite:
        shutil.rmtree(base_path, ignore_errors=True)
        os.makedirs(base_path, exist_ok=True)

    # Make Logger and Callbacks
    logger = pl_loggers.CSVLogger(save_dir=os.path.join(base_path, "logs"), name=experiment_name)
    early_stopping_callback, checkpoint_callback = _make_callbacks(base_path, experiment_name)

    # Prepare Data
    dataset = _load_dataset(language, data_path)
    dataset.prepare_data()
    dataset.setup(stage="fit")

    # Make Model and Trainer
    model = _make_model(dataset, hyperparameters, num_symbol_features, num_source_features, autoregressive_order)
    trainer = Trainer(
        max_epochs=500, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
        devices=1, gradient_clip_val=1.0, enable_progress_bar=verbose, logger=logger,
        enable_model_summary=verbose, callbacks=[early_stopping_callback, checkpoint_callback],
    )

    # Train Model and Load Best Checkpoint
    trainer.fit(model=model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())
    model.load_from_checkpoint(checkpoint_path=os.path.join(base_path, "saved_models", "last.ckpt"))

    logs = pd.read_csv(os.path.join(base_path, "logs", experiment_name, "version_0", "metrics.csv"))
    best_val_score = logs["val_normalised_edit_distance"].min()
    best_val_score = 100 * best_val_score

    # Get Predictions (optional)
    if get_predictions:
        predictions = predict(trainer, model, dataset)
    else:
        predictions = None

    return {"best_val_score": best_val_score, "predictions": predictions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Tuning")
    parser.add_argument("--basepath", default="./results")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--language", type=str)
    parser.add_argument("--symbol_features", type=int, default=0)
    parser.add_argument("--source_features", type=int, default=0)
    parser.add_argument("--autoregressive_order", type=int, default=0)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--layers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--hidden", type=int, default=256),
    parser.add_argument("--dropout", type=float, default=0.0),
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    hyper_parameters = Hyperparameters(
        batch_size=args.batch, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout,
        scheduler_gamma=args.gamma
    )

    result = experiment(
        base_path=args.basepath, data_path=args.datapath, language=args.language,
        num_source_features=args.source_features, num_symbol_features=args.symbol_features,
        autoregressive_order=args.autoregressive_order, overwrite=True, get_predictions=True, verbose=True,
        hyperparameters=hyper_parameters
    )

    print(f"\n\nBest Validation Score:\t {result['best_val_score']:.2f}\n\n")

    predictions_file_name = args.language
    predictions_file_name = predictions_file_name + f"num_source_features={args.source_features}"
    predictions_file_name = predictions_file_name + f"num_symbol_features={args.symbol_features}"
    predictions_file_name = predictions_file_name + f"autoregressive_order={args.autoregressive_order}"
    predictions_file_name = predictions_file_name + f"trial={args.trial}"
    predictions_file_name = predictions_file_name + ".json"

    os.makedirs("./predictions", exist_ok=True)
    with open(os.path.join("./predictions", predictions_file_name), "w") as psf:
        json.dump(result["predictions"], psf)
