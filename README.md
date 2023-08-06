# Tü-CL at SIGMORPHON 2023: Straight-Through Gradient Estimation for Hard Attention
## Introduction

This repository contains code for our system description paper [Tü-CL at SIGMORPHON 2023: Straight-Through Gradient Estimation for Hard Attention](https://aclanthology.org/2023.sigmorphon-1.17/) for the [Morphological Inflection shared task](https://aclanthology.org/2023.sigmorphon-1.13/).
Note, that the code for the interlinear glossing model can be found [in this repository](https://github.com/LGirrbach/sigmorphon-2023-glossing).

The implementation of the hard-attention transducer is in [model.py](model.py).

## Setup
Create a virtual environment, e.g. by using [Anaconda](https://docs.conda.io/en/latest/miniconda.html):
```
conda create -n inflection python=3.9 pip
```
Activate the environment:
```
conda activate inflection
```
Then, install the dependencies in [requirements.txt](requirements.txt):
```
pip install -r requirements.txt
```
Finally, place the shared task data in the repository, i.e. there should be a folder called `data`. The data can be obtained from [the shared task's main repository](https://github.com/sigmorphon/2023InflectionST/tree/main/part1).

## Train a model
To train a single model and get predictions for the corresponding test set, run
```
python experiment.py --language LANGUAGE
```
Here, `LANGUAGE` is a (three letter) code for the language. We assume the respective files `LANGUAGE.trn`, `LANGUAGE.dev` and `LANGUAGE.covered.tst` are in `./data`.
To see all command line options and hyperparameters, run
```
python experiment.py --help
```

## Hyperparameter tuning
To tune hyperparameters, use the script [hyperparameter_tuning.py](hyperparameter_tuning.py).
The main parameters are `--language` which specifies the dataset as before, and `--trials`, which specifies the number of evaluated hyperparameter combinations.
To parse the results from tuning logs, we provide the scrip [parse_hyperparameters.py](parse_hyperparameters.py).
The best parameters from our tuning runs are in [best_hyperparameters.json](best_hyperparameters.json).
We evaluated 50 trials per language to get these parameters.

Finally, we provide a script [train_best_parameters.py](train_best_parameters.py), that retrains models (from scratch) using the hyperparameters in [best_hyperparameters.json](best_hyperparameters.json).

## Citation
If you use this code, consider citing our paper:
```
@inproceedings{girrbach-2023-tu,
    title = {T{\"u}-{CL} at {SIGMORPHON} 2023: Straight-Through Gradient Estimation for Hard Attention},
    author = "Girrbach, Leander",
    booktitle = "Proceedings of the 20th SIGMORPHON workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sigmorphon-1.17",
    pages = "151--165",
}
```






