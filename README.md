# Are We Forgetting Something?

## Installing required components
Requirements for the experiments are specified in the `requirements.txt` file, and can be installed using pip.

```
pip install -r requirements.txt
```

## Running an experiment
To run an experiment, use the `run.py` file. This file uses click to allow command line parameters to be specified.

```
python run.py --dataset <dataset> --algorithm <algorithm> --timeout <timeout in seconds>
```

This will run the experiment for the specified dataset and algorithm, optimising parameters for the specified timeout in seconds.

Supported values for `<dataset>` and `<algorithm>` are given below.

### Supported datasets

* adressa
* recsys2015
* cosmeticsshop
* Amazon games
* Amazon Toys and Games

### Supported algorithms

* Popularity
* ItemKNN
* EASE
* TARSItemKNNLiu
* TARSItemKNNDing
* SequentialRules
* GRU4RecNegSampling

### Reproducing the experiments in the paper
In order to reproduce the experiments in the paper, for each dataset use the `runall_<dataset>.sh` bash script. This will run the experiment for each algorithm, with the preselected timeouts.

## Rendering results
Results are stored in the `results_hyperopt/<dataset>` folders. There is a CSV file for the optimisation and evaluation results for each algorithm.
To avoid inspecting these files one by one, we provide a notebook that generates the tables.
Follow the steps in the `Results.ipynb` notebook to plot and generate the result tables.