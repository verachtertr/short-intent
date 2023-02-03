import click
import datetime
from hyperopt import hp, Trials, fmin, tpe, space_eval, STATUS_OK
import numpy as np
from pathlib import Path

from recpack.datasets import AdressaOneWeek, CosmeticsShop, RecsysChallenge2015
from recpack.pipelines import PipelineBuilder
from recpack.scenarios import TimedLastItemPrediction, Timed
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem

import pandas as pd
import time

from amazon_dataset import AmazonGamesDataset, AmazonToysAndGamesDataset

DATASET_PATH = "/home/robinverachtert/datasets/"
HOUR = 3600
DAY = 24 * 3600
WEEK = 7 * DAY
MONTH = 31 * DAY
YEAR = 365 * DAY


def recsys_dataset(dataset_path):
    ds = RecsysChallenge2015(
        path=dataset_path, filename="yoochoose-clicks.dat", use_default_filters=False
    )
    ds.add_filter(MinUsersPerItem(50, item_ix=ds.ITEM_IX, user_ix=ds.USER_IX))
    ds.add_filter(MinItemsPerUser(3, item_ix=ds.ITEM_IX, user_ix=ds.USER_IX))

    return ds


def get_datasets_info(dataset_path, dataset):
    datasets = {
        "adressa": {
            "dataset": AdressaOneWeek(path=dataset_path),
            "t": int(datetime.datetime(2017, 1, 7, 12).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 1, 6, 12).strftime("%s")),
            "delta_out": 12 * 3600,
            "delta_space": hp.loguniform("delta", np.log(1800), np.log(7 * 24 * 3600)),
        },
        "cosmeticsshop": {
            "dataset": CosmeticsShop(
                path=dataset_path, filename="cosmeticsshop_views.csv"
            ),
            "t": int(datetime.datetime(2020, 2, 15, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2020, 2, 1, 0).strftime("%s")),
            "delta_out": 14 * 24 * 3600,
            "delta_space": hp.loguniform(
                "delta", np.log(1800), np.log(152 * 24 * 3600)
            ),
        },
        "recsys2015": {
            "dataset": recsys_dataset(dataset_path),
            "t": int(datetime.datetime(2014, 9, 15, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2014, 9, 1, 0).strftime("%s")),
            "delta_out": 14 * 24 * 3600,
            "delta_space": hp.loguniform(
                "delta", np.log(1800), np.log(180 * 24 * 3600)
            ),
        },
        "amazon_games": {
            "dataset": AmazonGamesDataset(dataset_path),
            "t": int(datetime.datetime(2018, 4, 1, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 10, 1, 0).strftime("%s")),
            "delta_out": 6 * 31 * 24 * 3600,
            "delta_space": hp.loguniform("delta", np.log(MONTH), np.log(20 * YEAR)),
        },
        "amazon_toys_and_games": {
            "dataset": AmazonToysAndGamesDataset(dataset_path),
            "t": int(datetime.datetime(2018, 4, 1, 0).strftime("%s")),
            "t_val": int(datetime.datetime(2017, 10, 1, 0).strftime("%s")),
            "delta_out": 6 * 31 * 24 * 3600,
            "delta_space": hp.loguniform("delta", np.log(MONTH), np.log(20 * YEAR)),
        },
    }
    return datasets[dataset]


algorithms = {
    "ItemKNN": {
        "hyperopt_space": {
            "K": hp.uniformint("K", 50, 1000),
            "normalize_X": hp.choice("normalize_X", [True, False]),
            "similarity": hp.choice(
                "similarity", ["cosine", "conditional_probability"]
            ),
        },
    },
    "Popularity": {
        "hyperopt_space": {},
    },
    "EASE": {
        "hyperopt_space": {
            "l2": hp.uniform("l2", 1, 2000),
        },
        "fixed_params": {"density": 0.05},
    },
    "TARSItemKNNLiu": {
        "hyperopt_space": {
            "K": hp.uniformint("K", 50, 1000),
            "fit_decay": hp.loguniform("fit_decay", np.log(1e-10), np.log(1)),
            "predict_decay": hp.loguniform("predict_decay", np.log(1e-10), np.log(1)),
        },
    },
    "TARSItemKNNDing": {
        "hyperopt_space": {
            "K": hp.uniformint("K", 50, 1000),
            "predict_decay": hp.loguniform("predict_decay", np.log(1e-10), np.log(1)),
            "similarity": hp.choice(
                "similarity", ["cosine", "conditional_probability"]
            ),
        },
    },
    "TARSItemKNNXia": {
        "hyperopt_space": {
            "fit_decay": hp.uniform("fit_decay", 0, 1),
            "K": hp.uniformint("K", 50, 1000),
            "decay_interval": hp.uniformint("decay_interval", 60, 24 * 3600),
        }
    },
    "SequentialRules": {
        "hyperopt_space": {
            "K": hp.uniformint("K", 50, 1000),
            "max_steps": hp.uniformint("max_steps", 1, 100),
        },
    },
    "STAN": {
        "hyperopt_space": {
            "K": hp.uniformint("K", 50, 1000),
            "interaction_decay": hp.loguniform(
                "interaction_decay", np.log(1e-10), np.log(1)
            ),
            "session_decay": hp.loguniform("session_decay", np.log(1e-10), np.log(1)),
            "distance_from_match_decay": hp.loguniform(
                "distance_from_match_decay", np.log(0.01), np.log(10)
            ),
        },
    },
    "GRU4RecNegSampling": {
        "hyperopt_space": {
            "bptt": hp.uniformint("bptt", 1, 10),
            # 'learning_rate': hp.uniform('learning_rate', 1e-5, 0.01),
            "hidden_size": hp.uniformint("hidden_size", 50, 250),
            "embedding_size": hp.uniformint("embedding_size", 50, 300),
        },
        "fixed_params": {
            "stopping_criterion": "ndcg",
            "batch_size": 512,
            "max_epochs": 8,
            "predict_topK": 100,
            "validation_sample_size": 10000,
            "keep_last": True,
            "learning_rate": 0.1,
            "num_layers": 1,
        },
        "test_params": {"keep_last": True},
    },
}


@click.command()
@click.option("--dataset", help="Dataset to use for running the experiment.")
@click.option("--dataset-path", help="path to the dataset files", default=DATASET_PATH)
@click.option("--algorithm", help="The algorithm to run.")
@click.option("--timeout", help="Optimisation timeout", default=3600)
@click.option("--scenario-name", default="Timed")
def run(dataset, dataset_path, algorithm, timeout, scenario_name):
    if scenario_name not in ["Timed", "TimedLastItemPrediction"]:
        raise ValueError(f"{scenario_name} not supported.")

    print(f"running {algorithm} on {dataset}, with {scenario_name} scenario")

    path_to_results = f"results_{scenario_name}/{dataset}"

    print(">> Loading dataset")
    im = get_datasets_info(dataset_path, dataset)["dataset"].load()
    print("<< Loaded dataset")
    print(f"dataset shape = {im.shape}")

    t = get_datasets_info(dataset_path, dataset)["t"]
    t_val = get_datasets_info(dataset_path, dataset)["t_val"]
    delta_out = get_datasets_info(dataset_path, dataset)["delta_out"]

    print(">> Splitting dataset")
    if scenario_name == "Timed":
        scenario = Timed(t=t, t_validation=t_val, validation=True, delta_out=delta_out)
    elif scenario_name == "TimedLastItemPrediction":
        scenario = TimedLastItemPrediction(
            t=t, t_validation=t_val, validation=True, delta_out=delta_out
        )
    scenario.split(im)

    print("<< Split dataset")

    def evaluate(params):
        start = time.time()
        delta = params.pop("delta")
        pb_opt = PipelineBuilder()

        pb_opt.add_algorithm(
            algorithm,
            params={**params, **algorithms[algorithm].get("fixed_params", {})},
        )
        pb_opt.add_metric("NDCGK", K=10)

        pb_opt.set_test_data(scenario.validation_data)
        pb_opt.set_validation_data(scenario.validation_data)

        pb_opt.set_full_training_data(
            scenario.validation_training_data.timestamps_gt(t_val - delta)
        )
        pb_opt.set_validation_training_data(
            scenario.validation_training_data.timestamps_gt(t_val - delta)
        )

        pipe = pb_opt.build()
        pipe.run()
        end = time.time()

        # returning negative ndcg as loss, since hyperopt minimises things
        return {
            "loss": -pipe.get_metrics()["NDCGK_10"].values[0],
            "run_time": end - start,
            "status": STATUS_OK,
        }

    tpe_trials = Trials()

    space = algorithms[algorithm]["hyperopt_space"].copy()
    space["delta"] = get_datasets_info(dataset_path, dataset)["delta_space"]

    best = fmin(evaluate, space, algo=tpe.suggest, trials=tpe_trials, timeout=timeout)

    optimal_params = space_eval(space, best)
    optimal_delta = optimal_params.pop("delta")

    optim_results = {
        "loss": [x["loss"] for x in tpe_trials.results],
        "run_time": [x["run_time"] for x in tpe_trials.results],
        "iteration": tpe_trials.idxs_vals[0]["delta"],
    }
    for key in space:
        optim_results[key] = tpe_trials.idxs_vals[1][key]
    tpe_results = pd.DataFrame(optim_results)

    tpe_results["ndcg"] = -tpe_results["loss"]
    tpe_results["window_hours"] = tpe_results["delta"] / 3600

    p1 = Path(f"results_{scenario_name}")
    p1.mkdir(exist_ok=True)
    p2 = Path(path_to_results)
    p2.mkdir(exist_ok=True)
    tpe_results.to_csv(
        f"{path_to_results}/optimisation_results_{algorithm}.csv",
        header=True,
        index=False,
    )

    optimal_params = {
        **optimal_params,
        **algorithms[algorithm].get("fixed_params", {}),
        **algorithms[algorithm].get("test_params", {}),
    }
    print(f"optimal params = {optimal_params}")
    print(f"optimal_delta = {optimal_delta}")

    test_pipeline = PipelineBuilder()

    print(">> Running pipeline with full data")
    test_pipeline.add_algorithm(algorithm, params=optimal_params)
    test_pipeline.add_metric("CoverageK", K=[10, 20, 50])
    test_pipeline.add_metric("NDCGK", K=[10, 20, 50])
    test_pipeline.add_metric("CalibratedRecallK", K=[10, 20, 50])
    test_pipeline.add_metric("ReciprocalRankK", K=[10, 20, 50])
    test_pipeline.add_metric("PrecisionK", K=[10, 20, 50])

    test_pipeline.set_test_data(scenario.test_data)
    test_pipeline.set_validation_data(scenario.validation_data)

    # We do some hacking here
    # By setting the full training data as validation training data,
    # we are sure that the model is trained on all data.
    # We'll keep the last model, rather than using the validation in and out data to select the model.
    test_pipeline.set_full_training_data(scenario.full_training_data)
    test_pipeline.set_validation_training_data(scenario.full_training_data)

    pipe = test_pipeline.build()
    pipe.run()

    maximal_result = pipe.get_metrics()

    print("<< Finished pipeline with full data")

    print(">> Running pipeline with optimal data")
    # Evaluate with optimal delta
    test_pipeline.set_full_training_data(
        scenario.full_training_data.timestamps_gt(t - optimal_delta)
    )
    test_pipeline.set_validation_training_data(
        scenario.full_training_data.timestamps_gt(t - optimal_delta)
    )

    pipe = test_pipeline.build()
    pipe.run()

    final_result = pipe.get_metrics()

    print("<< Finished pipeline with optimal data")
    final_result.to_csv(f"{path_to_results}/final_result_{algorithm}.csv")
    maximal_result.to_csv(f"{path_to_results}/full_window_result_{algorithm}.csv")


if __name__ == "__main__":
    run()
