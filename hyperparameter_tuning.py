import os
import sys
import optuna
import logging
import argparse

from experiment import experiment
from containers import Hyperparameters


def hyperparameter_tuning(
    base_path: str,
    data_path: str,
    model_type: str,
    language: str,
    num_symbol_features: int,
    num_source_features: int,
    autoregressive_order: int,
    num_trials: int = 1,
):
    # Define Optuna Logger
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Define Objective
    def objective(trial: optuna.Trial):
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        hidden_size = trial.suggest_int("hidden_size", 64, 512)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.9, 1.0)
        batch_size = trial.suggest_int("batch_size", 4, 64)

        hyperparameters = Hyperparameters(
            batch_size=batch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            scheduler_gamma=scheduler_gamma,
        )
        result = experiment(
            base_path=base_path,
            data_path=data_path,
            model_type=model_type,
            language=language,
            hyperparameters=hyperparameters,
            num_source_features=num_source_features,
            num_symbol_features=num_symbol_features,
            autoregressive_order=autoregressive_order,
            overwrite=False,
            get_predictions=False,
        )
        return result["best_val_score"]

    # Setup Optuna
    os.makedirs("./tuning", exist_ok=True)
    study_name = f"inflection_tuning={language}"
    study_name = study_name + f"-model={model_type}"
    study_name = study_name + f"-num_symbol_features={num_symbol_features}"
    study_name = study_name + f"-num_source_features={num_source_features}"
    study_name = study_name + f"-autoregressive_order={autoregressive_order}"

    # Skip if exists
    if os.path.exists(f"./tuning/{study_name}.csv"):
        return
    elif os.path.exists(f"./tuning/{study_name}.db"):
        os.remove(f"./tuning/{study_name}.db")

    storage_name = f"sqlite:///tuning/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="minimize"
    )
    study.optimize(objective, n_trials=num_trials)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(f"./tuning/{study_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Tuning")
    parser.add_argument("--basepath", default="./tuning/results")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--language", type=str)
    parser.add_argument(
        "--model",
        type=str,
        choices=["interpretable", "seq2seq"],
        default="interpretable",
    )
    parser.add_argument("--symbol_features", type=int, default=0)
    parser.add_argument("--source_features", type=int, default=0)
    parser.add_argument("--autoregressive_order", type=int, default=0)
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()

    hyperparameter_tuning(
        base_path=args.basepath,
        data_path=args.datapath,
        model_type=args.model,
        language=args.language,
        num_trials=args.trials,
        num_source_features=args.source_features,
        num_symbol_features=args.symbol_features,
        autoregressive_order=args.autoregressive_order,
    )
