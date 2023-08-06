import json
import argparse

from typing import Dict
from experiment import experiment
from containers import Hyperparameters


def load_best_hyperparameters() -> Dict[str, any]:
    with open("./best_hyperparameters.json") as hf:
        return json.load(hf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inflection Experiment")
    parser.add_argument("--basepath", default="./retrain_results")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--language", type=str)
    parser.add_argument(
        "--model",
        type=str,
        choices=["interpretable", "seq2seq"],
        default="interpretable",
    )
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--symbol_features", type=int, default=0)
    parser.add_argument("--source_features", type=int, default=0)
    parser.add_argument("--autoregressive_order", type=int, default=0)
    args = parser.parse_args()

    best_hyperparameters = load_best_hyperparameters()
    language_parameters = best_hyperparameters[args.language]

    hyper_parameters = Hyperparameters(
        batch_size=int(language_parameters["batch_size"]),
        hidden_size=int(language_parameters["hidden_size"]),
        num_layers=int(language_parameters["num_layers"]),
        dropout=language_parameters["dropout"],
        scheduler_gamma=language_parameters["scheduler_gamma"],
    )

    result = experiment(
        base_path=args.basepath,
        data_path=args.datapath,
        language=args.language,
        model_type=args.model,
        num_source_features=args.source_features,
        num_symbol_features=args.symbol_features,
        autoregressive_order=args.autoregressive_order,
        overwrite=True,
        get_predictions=False,
        verbose=False,
        hyperparameters=hyper_parameters,
        trial=args.trial,
    )
