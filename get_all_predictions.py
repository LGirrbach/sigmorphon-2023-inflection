import os
import torch
import pickle

from predict import predict
from baseline import Seq2SeqModel
from experiment import _load_dataset
from pytorch_lightning import Trainer
from model import InterpretableTransducer


if __name__ == "__main__":
    os.makedirs("./predictions", exist_ok=True)

    for model_name in sorted(os.listdir("./retrain_results")):
        language = model_name.split("-")[0].strip()
        parameters = {
            key: value
            for key, value in [entry.split("=") for entry in model_name.split("-")[1:]]
        }
        if "model" not in parameters:
            parameters["model"] = "interpretable"

        saved_model_path = os.path.join("./retrain_results", model_name, "saved_models")
        saved_model_filename = [
            filename
            for filename in os.listdir(saved_model_path)
            if filename != "last.ckpt"
        ]
        saved_model_filename = saved_model_filename[0]
        saved_model_path = os.path.join(saved_model_path, saved_model_filename)

        dataset = _load_dataset(language, data_path="./data")
        dataset.prepare_data()
        dataset.setup(stage="fit")
        dataset.setup(stage="test")

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=True,
            logger=False,
            enable_model_summary=False,
        )

        if parameters["model"] == "interpretable":
            model = InterpretableTransducer.load_from_checkpoint(saved_model_path)
        elif parameters["model"] == "seq2seq":
            model = Seq2SeqModel.load_from_checkpoint(saved_model_path)
        else:
            raise ValueError(f"Unknown model type: {parameters['model']}")

        model_predictions = predict(trainer=trainer, model=model, dataset=dataset)

        with open(
            os.path.join(
                "./predictions",
                f"language={language}-model={parameters['model']}-trial={parameters['trial']}.pickle",
            ),
            "wb",
        ) as sf:
            pickle.dump(model_predictions, sf)
