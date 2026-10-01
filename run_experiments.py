#!/usr/bin/env python3
"""CPU-conscious experiment runner for the workload forecasting models."""

import argparse
import ast
import csv
from contextlib import contextmanager
import datetime
import glob
import logging
import math
import os
import statistics
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("quick", "model_comparison", "transfer_learning", "runtime", "full"),
        default="quick",
    )
    parser.add_argument("--runs", type=int, default=3, help="Repeated random seeds (default: 3)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", choices=("LSTM", "HBNN", "LSTMD"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=5)
    parser.add_argument("--inference-repeats", type=int, default=10)
    parser.add_argument("--max-transfer-targets", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def replace_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


@contextmanager
def keras_python312_random_compatibility():
    """Adapt tf-keras 2.16's float randint bound to Python 3.12 temporarily."""
    import random

    original_randint = random.randint

    def compatible_randint(lower, upper):
        if isinstance(upper, float) and upper.is_integer():
            upper = int(upper)
        return original_randint(lower, upper)

    random.randint = compatible_randint
    try:
        yield
    finally:
        random.randint = original_randint


def _csv_value(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return ""
    except TypeError:
        pass
    return value


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def timestamp_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def dataset_group(name):
    if name.startswith("gc19_"):
        return "Google-2019"
    if name == "gc11":
        return "Google-2011"
    if name == "ali18":
        return "Alibaba-2018"
    if name.startswith("ali20_"):
        return "Alibaba-2020"
    return "unknown"


def make_run_id(invocation_id, model, dataset, seed):
    return "{}-{}-{}-seed{}".format(invocation_id, model, dataset, seed)


def default_parameters(model_name):
    return {
        "first_conv_dim": 16,
        "first_conv_kernel": 3,
        "first_conv_activation": "relu",
        "cnn_layers": 1,
        "second_lstm_dim": 32,
        "first_dense_dim": 16,
        "first_dense_activation": "relu",
        "mlp_units": [32],
        "dense_kernel_init": "glorot_uniform",
        "batch_size": 64,
        "epochs": 30,
        "patience": 8,
        "optimizer": "adam",
        "lr": 0.001,
        "momentum": 0.9,
        "decay": 0.0,
    }


def parameter_candidates(model_name, dataset, prediction_type, training_type, resource):
    if prediction_type == "bivariate":
        folder = "multibivariate" if training_type == "multiple" else "bivariate"
        prefix = {"LSTM": "BILSTM", "HBNN": "BIHBNN", "LSTMD": "BILSTMD"}[model_name]
        names = [
            "{}-{}-w288-h2.csv".format(prefix, dataset),
            "{}-w288-h2.csv".format(model_name),
            "{}-w288-h2.csv".format(prefix),
        ]
    elif training_type == "multiple":
        folder = "multiunivariate"
        names = [
            "{}-{}-w288-h2.csv".format(model_name, resource),
            "{}-{}-w288-h2.csv".format(model_name, "cpu"),
        ]
    else:
        folder = "univariate"
        names = [
            "{}-{}-{}-w288-h2.csv".format(model_name, dataset, resource),
            "{}-{}-cpu-w288-h2.csv".format(model_name, dataset),
            "{}-{}-mem-w288-h2.csv".format(model_name, dataset),
        ]
    return [os.path.join(ROOT, "hyperparams", folder, name) for name in names]


def load_parameters(model_name, dataset, prediction_type, training_type, resource, args):
    import numpy as np

    parameters = default_parameters(model_name)
    parameter_path = next(
        (candidate for candidate in parameter_candidates(
            model_name, dataset, prediction_type, training_type, resource
        ) if os.path.isfile(candidate)),
        None,
    )
    if parameter_path:
        with open(parameter_path, "r", newline="", encoding="utf-8") as source:
            row = next(csv.DictReader(source), {})
        for key in parameters:
            value = row.get(key)
            if value in (None, ""):
                continue
            if key == "mlp_units":
                try:
                    parameters[key] = [int(unit) for unit in ast.literal_eval(value)]
                except (ValueError, SyntaxError, TypeError):
                    parameters[key] = []
            elif key in ("first_conv_dim", "first_conv_kernel", "cnn_layers",
                         "second_lstm_dim", "first_dense_dim", "batch_size",
                         "epochs", "patience"):
                parameters[key] = int(float(value))
            elif key in ("lr", "momentum", "decay"):
                parameters[key] = float(value)
            elif key == "first_dense_activation":
                parameters[key] = "tanh" if "tanh" in value.lower() else "relu"
            else:
                parameters[key] = value
    if args.epochs is not None:
        parameters["epochs"] = args.epochs
    elif args.mode == "quick":
        parameters["epochs"] = min(parameters["epochs"], 3)
    else:
        parameters["epochs"] = min(parameters["epochs"], 100)
    if args.patience is not None:
        parameters["patience"] = args.patience
    else:
        parameters["patience"] = min(parameters["patience"], 8 if args.mode != "quick" else 2)
    if args.batch_size is not None:
        parameters["batch_size"] = args.batch_size
    else:
        parameters["batch_size"] = min(parameters["batch_size"], 64 if args.mode == "quick" else 128)
    parameters["weight_file"] = ""
    parameters["_parameter_path"] = parameter_path or "defaults"
    return parameters


def make_model(model_name):
    if model_name == "LSTM":
        from models.LSTM import LSTMPredictor
        return LSTMPredictor()
    if model_name == "HBNN":
        from models.HBNN import HBNNPredictor
        return HBNNPredictor()
    from models.LSTMD import LSTMDPredictor
    return LSTMDPredictor()


def set_seed(seed):
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass


def prepare_split(dataset, resources, args, quick_limits=True):
    from util.experiment_data import load_dataset_split

    train_limit = args.max_train_windows
    test_limit = args.max_test_windows
    if args.mode == "quick" and quick_limits:
        train_limit = train_limit or 512
        test_limit = test_limit or 256
    return load_dataset_split(
        dataset,
        resources,
        window_size=args.window_size,
        horizon_steps=args.horizon,
        train_fraction=0.8,
        validation_fraction=0.2,
        max_train_windows=train_limit,
        max_test_windows=test_limit,
    )


def stack_splits(splits):
    import numpy as np

    if len(splits) == 1:
        split = splits[0]
        return {key: split[key] for key in ("X_train", "y_train", "X_val", "y_val")}
    return {
        key: np.concatenate([split[key] for split in splits], axis=0)
        for key in ("X_train", "y_train", "X_val", "y_val")
    }


def predict_values(model, model_name, inputs, mc_samples):
    import numpy as np

    if model_name == "LSTM":
        return np.asarray(model.predict(inputs, verbose=0), dtype=np.float32), None, None, None
    if model_name == "LSTMD":
        distribution = model(inputs, training=False)
        mean = np.asarray(distribution.mean().numpy(), dtype=np.float32)
        standard_deviation = np.asarray(distribution.stddev().numpy(), dtype=np.float32)
        return mean, standard_deviation, None, standard_deviation

    sampled_means = []
    sampled_variances = []
    for _ in range(max(1, mc_samples)):
        distribution = model(inputs, training=False)
        sampled_means.append(np.asarray(distribution.mean().numpy(), dtype=np.float32))
        sampled_variances.append(
            np.square(np.asarray(distribution.stddev().numpy(), dtype=np.float32))
        )
    sampled_means = np.stack(sampled_means, axis=0)
    epistemic_variance = np.var(sampled_means, axis=0)
    aleatoric_variance = np.mean(np.stack(sampled_variances, axis=0), axis=0)
    mean = np.mean(sampled_means, axis=0)
    epistemic_std = np.sqrt(np.maximum(epistemic_variance, 0.0))
    aleatoric_std = np.sqrt(np.maximum(aleatoric_variance, 0.0))
    total_std = np.sqrt(np.maximum(epistemic_variance + aleatoric_variance, 0.0))
    return mean, total_std, epistemic_std, aleatoric_std


def predict_in_chunks(model, model_name, inputs, mc_samples, batch_size):
    import numpy as np

    means = []
    stds = []
    epistemic = []
    aleatoric = []
    for start in range(0, len(inputs), batch_size):
        mean, std, epistemic_std, aleatoric_std = predict_values(
            model, model_name, inputs[start:start + batch_size], mc_samples
        )
        means.append(mean)
        if std is not None:
            stds.append(std)
        if epistemic_std is not None:
            epistemic.append(epistemic_std)
        if aleatoric_std is not None:
            aleatoric.append(aleatoric_std)
    return (
        np.concatenate(means, axis=0),
        np.concatenate(stds, axis=0) if stds else None,
        np.concatenate(epistemic, axis=0) if epistemic else None,
        np.concatenate(aleatoric, axis=0) if aleatoric else None,
    )


def current_peak_memory_mb():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(peak / 1024.0)
    except (ImportError, AttributeError):
        return float("nan")


def model_size_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(root, name))
                   for root, _, files in os.walk(path) for name in files)
    return 0


def measure_single_sample_latency(model, model_name, sample, repeats):
    durations = []
    predict_values(model, model_name, sample, 1)
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        predict_values(model, model_name, sample, 1)
        durations.append(time.perf_counter() - start)
    return {
        "inference_time": statistics.mean(durations),
        "inference_time_median": statistics.median(durations),
        "inference_time_std": statistics.pstdev(durations),
    }


def row_metadata(invocation_id, seed, model_label, dataset_label,
                 prediction_type, training_type, resource_name, run_id):
    return {
        "timestamp": invocation_id,
        "seed": seed,
        "model": model_label,
        "dataset": dataset_label,
        "prediction_type": prediction_type,
        "training_type": training_type,
        "resource": resource_name,
        "run_id": run_id,
    }


def train_model_run(model_name, model_label, dataset_label, prediction_type,
                    training_type, source_splits, target_split, args,
                    invocation_id, seed, fine_tune=False):
    import numpy as np
    import tensorflow as tf
    from util.experiment_metrics import (
        DEFAULT_QUANTILES,
        DEFAULT_SERVICE_LEVELS,
        gaussian_quantile,
        point_metrics,
        probabilistic_metrics,
        residual_quantile_bounds,
    )

    set_seed(seed)
    tf.keras.backend.clear_session()
    resources = target_split["resources"]
    resource_name_for_params = "cpu" if resources[0] == "avgcpu" else "mem"
    params = load_parameters(
        model_name, target_split["dataset"], prediction_type, training_type,
        resource_name_for_params, args
    )
    train_data = stack_splits(source_splits)
    predictor = make_model(model_name)
    predictor.name = make_run_id(invocation_id, model_label, dataset_label, seed)
    checkpoint_dir = os.path.join(ROOT, "results", "raw", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    predictor.model_path = checkpoint_dir + os.sep

    start_training = time.perf_counter()
    with keras_python312_random_compatibility():
        _, trained_model = predictor.talos_model(
            train_data["X_train"], train_data["y_train"],
            train_data["X_val"], train_data["y_val"], params,
        )
    training_seconds = time.perf_counter() - start_training
    predictor.model = trained_model
    predictor.train_model = trained_model

    fine_tuning_seconds = 0.0
    checkpoint_path = os.path.join(checkpoint_dir, predictor.name + ".weights.h5")
    trained_model.save_weights(checkpoint_path)
    if fine_tune:
        trained_model.load_weights(checkpoint_path)
        start_fine_tuning = time.perf_counter()
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=params["patience"],
            restore_best_weights=True,
        )
        trained_model.fit(
            target_split["X_train"], target_split["y_train"],
            validation_data=(target_split["X_val"], target_split["y_val"]),
            epochs=params["epochs"], batch_size=params["batch_size"],
            verbose=0, callbacks=[early_stopping],
        )
        fine_tuning_seconds = time.perf_counter() - start_fine_tuning
        trained_model.save_weights(checkpoint_path)

    batch_size = max(1, min(params["batch_size"], 256))
    test_mean, test_std, test_epistemic, test_aleatoric = predict_in_chunks(
        trained_model, model_name, target_split["X_test"], args.mc_samples, batch_size
    )
    val_mean, _, _, _ = predict_in_chunks(
        trained_model, model_name, target_split["X_val"], args.mc_samples, batch_size
    )
    point_rows = []
    quantile_rows = []
    service_rows = []
    uncertainty_rows = []
    uncertainty_summary_rows = []
    prediction_rows = []
    runtime_latency = measure_single_sample_latency(
        trained_model, model_name, target_split["X_test"][:1], args.inference_repeats
    )
    val_residuals = target_split["y_val"] - val_mean

    for channel, resource in enumerate(resources):
        actual = target_split["y_test"][:, channel:channel + 1]
        mean = test_mean[:, channel:channel + 1]
        metric_values = point_metrics(actual, mean)
        one_channel_std = test_std[:, channel:channel + 1] if test_std is not None else None
        if one_channel_std is not None:
            channel_quantile_rows, channel_service_rows = probabilistic_metrics(
                actual, mean, one_channel_std,
                service_levels=DEFAULT_SERVICE_LEVELS,
                quantile_levels=DEFAULT_QUANTILES,
            )
            for row in channel_quantile_rows:
                quantile_rows.append({"resource": resource, **row})
            for row in channel_service_rows:
                service_rows.append({"resource": resource, **row})
            ql_99 = next((row["QL"] for row in channel_quantile_rows
                          if abs(row["quantile"] - 0.99) < 1e-8), None)
            central_lower = gaussian_quantile(mean, one_channel_std, 0.025)
            central_upper = gaussian_quantile(mean, one_channel_std, 0.975)
        else:
            ql_99 = None
            calibrated = residual_quantile_bounds(
                target_split["y_val"][:, channel:channel + 1],
                val_mean[:, channel:channel + 1], actual, mean,
                service_levels=(0.95,),
            )[0.95]
            central_lower, central_upper = calibrated
            for service_level in DEFAULT_SERVICE_LEVELS:
                upper_offset = np.quantile(val_residuals[:, channel], service_level)
                upper = mean + upper_offset
                demand_sum = float(np.sum(actual))
                success_rate = float(np.mean(actual <= upper) * 100.0)
                service_rows.append({
                    "resource": resource,
                    "service_level": float(service_level),
                    "quantile": float(service_level),
                    "SR": success_rate,
                    "TPR": float(100.0 * np.sum(upper) / demand_sum) if demand_sum > 0 else float("nan"),
                    "coverage": float(np.mean((actual >= central_lower) & (actual <= central_upper)) * 100.0),
                    "calibration_error": success_rate - service_level * 100.0,
                    "mean_interval_width": float(np.mean(central_upper - central_lower)),
                })

        common = row_metadata(
            invocation_id, seed, model_label, dataset_label,
            prediction_type, training_type, resource,
            predictor.name,
        )
        if one_channel_std is not None:
            uncertainty_summary_rows.append({
                **common,
                "summary_scope": "test_run",
                "mean_prediction": float(np.mean(mean)),
                "prediction_std_mean": float(np.mean(one_channel_std)),
                "prediction_std_median": float(np.median(one_channel_std)),
                "prediction_std_std": float(np.std(one_channel_std)),
                "epistemic_uncertainty": (
                    float(np.mean(test_epistemic[:, channel]))
                    if test_epistemic is not None else ""
                ),
                "aleatoric_uncertainty": (
                    float(np.mean(test_aleatoric[:, channel]))
                    if test_aleatoric is not None else float(np.mean(one_channel_std))
                ),
                "total_uncertainty": float(np.mean(one_channel_std)),
            })
        point_rows.append({
            **common,
            **metric_values,
            "QL": ql_99,
            "training_time": training_seconds,
            "fine_tuning_time": fine_tuning_seconds,
            **runtime_latency,
            "number_of_parameters": int(trained_model.count_params()),
            "model_size_bytes": model_size_bytes(checkpoint_path),
            "peak_memory_mb": current_peak_memory_mb(),
            "epochs_requested": params["epochs"],
            "batch_size": params["batch_size"],
            "window_size": args.window_size,
            "horizon_steps": args.horizon,
            "parameter_file": params["_parameter_path"],
        })

        if one_channel_std is not None:
            epistemic_channel = (
                test_epistemic[:, channel:channel + 1]
                if test_epistemic is not None else None
            )
            aleatoric_channel = (
                test_aleatoric[:, channel:channel + 1]
                if test_aleatoric is not None else one_channel_std
            )
            for index in range(len(actual)):
                uncertainty_rows.append({
                    **common,
                    "sample_index": index,
                    "timestamp_value": target_split["test_timestamps"][index],
                    "prediction_std": float(one_channel_std[index, 0]),
                    "epistemic_uncertainty": (
                        float(epistemic_channel[index, 0]) if epistemic_channel is not None else ""
                    ),
                    "aleatoric_uncertainty": float(aleatoric_channel[index, 0]),
                    "total_uncertainty": float(one_channel_std[index, 0]),
                })

        for index in range(len(actual)):
            prediction_rows.append({
                **common,
                "sample_index": index,
                "timestamp": target_split["test_timestamps"][index],
                "actual": float(actual[index, 0]),
                "prediction_mean": float(mean[index, 0]),
                "prediction_std": (
                    float(one_channel_std[index, 0]) if one_channel_std is not None else ""
                ),
                "lower_bound_95": float(central_lower[index, 0]),
                "upper_bound_95": float(central_upper[index, 0]),
                "epistemic_uncertainty": (
                    float(test_epistemic[index, channel]) if test_epistemic is not None else ""
                ),
                "aleatoric_uncertainty": (
                    float(test_aleatoric[index, channel]) if test_aleatoric is not None else ""
                ),
                "total_uncertainty": (
                    float(one_channel_std[index, 0]) if one_channel_std is not None else ""
                ),
            })

    runtime_row = {
        "timestamp": invocation_id,
        "seed": seed,
        "model": model_label,
        "dataset": dataset_label,
        "prediction_type": prediction_type,
        "training_type": training_type,
        "run_id": predictor.name,
        "training_time": training_seconds,
        "fine_tuning_time": fine_tuning_seconds,
        "inference_time": runtime_latency["inference_time"],
        "inference_time_median": runtime_latency["inference_time_median"],
        "inference_time_std": runtime_latency["inference_time_std"],
        "number_of_parameters": int(trained_model.count_params()),
        "model_size_bytes": model_size_bytes(checkpoint_path),
        "peak_memory_mb": current_peak_memory_mb(),
        "inference_repetitions": args.inference_repeats,
    }
    return (
        point_rows, quantile_rows, service_rows, uncertainty_rows,
        uncertainty_summary_rows, prediction_rows, runtime_row,
    )


def save_dataset_inventory(datasets, args, invocation_id):
    from util.experiment_data import load_dataset_split

    rows = []
    for dataset in datasets:
        split = load_dataset_split(
            dataset, ["avgcpu", "avgmem"], args.window_size, args.horizon,
            train_fraction=0.8, validation_fraction=0.2,
        )
        rows.append({
            "timestamp": invocation_id,
            "dataset": dataset,
            "provider": "Google" if dataset.startswith("gc") else "Alibaba",
            "trace": dataset_group(dataset),
            "number_of_samples": split["sample_count"],
            "number_of_features": split["feature_count"],
            "resources": "avgcpu;avgmem",
            "sampling_interval_seconds": split["sampling_interval_seconds"],
            "train_size": split["train_size"],
            "validation_size": split["validation_size"],
            "test_size": split["test_size"],
            "window_size": args.window_size,
            "horizon_steps": args.horizon,
        })
        del split
    append_csv(os.path.join(ROOT, "results", "metrics", "dataset_summary.csv"), rows)


def run_regular_task(model_name, dataset, prediction_type, training_type,
                     all_datasets, args, invocation_id, resource_name=None):
    from util.experiment_data import available_datasets

    resources = (
        ["avgcpu", "avgmem"] if prediction_type == "bivariate"
        else [resource_name or "avgcpu"]
    )
    source_names = [dataset] if training_type == "single" else all_datasets
    source_splits = [prepare_split(name, resources, args) for name in source_names]
    if training_type == "multiple":
        target_name = dataset if dataset in source_names else source_names[0]
    else:
        target_name = dataset
    target_split = next(split for split in source_splits if split["dataset"] == target_name)
    model_label = ("S" if training_type == "single" else "M") + "-" + (
        "U" if prediction_type == "univariate" else "B"
    ) + "-" + model_name
    dataset_label = target_name if training_type == "single" else "all"

    for run_index in range(args.runs):
        seed = args.seed + run_index
        outputs = train_model_run(
            model_name, model_label, dataset_label, prediction_type,
            training_type, source_splits, target_split, args, invocation_id, seed,
        )
        (point_rows, quantile_rows, service_rows, uncertainty_rows,
         uncertainty_summary_rows, prediction_rows, runtime_row) = outputs
        append_csv(os.path.join(ROOT, "results", "raw", "model_metrics.csv"), point_rows)
        append_csv(os.path.join(ROOT, "results", "raw", "quantile_metrics.csv"), [
            {"timestamp": invocation_id, "seed": seed, "model": model_label,
             "dataset": dataset_label, "prediction_type": prediction_type,
             "training_type": training_type, "run_id": point_rows[0]["run_id"], **row}
            for row in quantile_rows
        ])
        append_csv(os.path.join(ROOT, "results", "raw", "service_level_metrics.csv"), [
            {"timestamp": invocation_id, "seed": seed, "model": model_label,
             "dataset": dataset_label, "prediction_type": prediction_type,
             "training_type": training_type, "run_id": point_rows[0]["run_id"],
             "MSE": point_rows[0]["MSE"], "MAE": point_rows[0]["MAE"],
             "RMSE": point_rows[0]["RMSE"], **row}
            for row in service_rows
        ])
        append_csv(os.path.join(ROOT, "results", "raw", "uncertainty_metrics.csv"), uncertainty_rows)
        append_csv(os.path.join(ROOT, "results", "raw", "uncertainty_summary.csv"),
               uncertainty_summary_rows)
        append_csv(os.path.join(ROOT, "results", "raw", "runtime_metrics.csv"), [runtime_row])
        prediction_path = os.path.join(
            ROOT, "results", "predictions", "{}_{}.csv".format(model_label, point_rows[0]["run_id"])
        )
        append_csv(prediction_path, prediction_rows)
        print("{} {} seed {}: MSE {:.6f}, MAE {:.6f}".format(
            model_label, dataset_label, seed,
            statistics.mean(row["MSE"] for row in point_rows),
            statistics.mean(row["MAE"] for row in point_rows),
        ))
        del outputs
    del source_splits, target_split


def transfer_scenarios(datasets):
    google_2019 = [dataset for dataset in datasets if dataset.startswith("gc19_")]
    other = [dataset for dataset in datasets if dataset not in google_2019]
    scenarios = []
    for target in datasets:
        scenarios.append(("All", list(datasets), target, False))
        scenarios.append(("All-FT", list(datasets), target, True))
        scenarios.append(("All-but-one", [item for item in datasets if item != target], target, False))
        scenarios.append(("All-but-one-FT", [item for item in datasets if item != target], target, True))
        if target in google_2019 and len(google_2019) > 1:
            scenarios.append(("Same-distribution-ZS",
                              [item for item in google_2019 if item != target], target, False))
            scenarios.append(("Same-distribution-FT",
                              [item for item in google_2019 if item != target], target, True))
        elif target in other and google_2019:
            scenarios.append(("Different-distribution-ZS", list(google_2019), target, False))
            scenarios.append(("Different-distribution-FT", list(google_2019), target, True))
    return scenarios


def run_transfer(args, datasets, invocation_id):
    from util.experiment_data import load_dataset_split

    targets_seen = 0
    rows = []
    selected_scenarios = transfer_scenarios(datasets)
    for scenario, source_names, target_name, fine_tune in selected_scenarios:
        if args.max_transfer_targets is not None and target_name not in {
            item[2] for item in selected_scenarios[:args.max_transfer_targets]
        }:
            continue
        resources = ["avgcpu", "avgmem"]
        target_split = prepare_split(target_name, resources, args, quick_limits=False)
        source_splits = [prepare_split(name, resources, args, quick_limits=False)
                         for name in source_names]
        scenario_name = "TL-" + scenario
        model_label = "M-B-HBNN-" + scenario_name
        for run_index in range(args.runs):
            seed = args.seed + run_index
            outputs = train_model_run(
                "HBNN", model_label, target_name, "bivariate", "multiple",
                source_splits, target_split, args, invocation_id, seed,
                fine_tune=fine_tune,
            )
            point_rows, _, service_rows, _, _, _, runtime_row = outputs
            for point in point_rows:
                resource_rows = [row for row in service_rows if row["resource"] == point["resource"]]
                service_99 = next((row for row in resource_rows
                                   if abs(row["service_level"] - 0.99) < 1e-8), {})
                rows.append({
                    "timestamp": invocation_id,
                    "source_dataset": ";".join(source_names),
                    "target_dataset": target_name,
                    "source_group": ";".join(sorted(set(dataset_group(name) for name in source_names))),
                    "target_group": dataset_group(target_name),
                    "historical_data": bool(target_name in source_names or fine_tune),
                    "fine_tuning": bool(fine_tune),
                    "scenario": scenario,
                    "model": model_label,
                    "resource": point["resource"],
                    "seed": seed,
                    "run_id": point["run_id"],
                    "MSE": point["MSE"],
                    "MAE": point["MAE"],
                    "RMSE": point["RMSE"],
                    "QL": point["QL"],
                    "SR": service_99.get("SR"),
                    "TPR": service_99.get("TPR"),
                    "training_time": runtime_row["training_time"],
                    "fine_tuning_time": runtime_row["fine_tuning_time"],
                    "inference_time": runtime_row["inference_time"],
                })
            targets_seen += 1
            print("{} -> {}{} seed {}".format(
                ",".join(source_names), target_name, " fine-tuned" if fine_tune else " zero-shot", seed
            ))
        del source_splits, target_split
    append_csv(os.path.join(ROOT, "results", "raw", "transfer_learning_metrics.csv"), rows)


def aggregate_metrics():
    numeric_fields = ("MSE", "MAE", "RMSE", "R2", "QL", "training_time",
                      "fine_tuning_time", "inference_time", "number_of_parameters",
                      "model_size_bytes", "peak_memory_mb")
    rows = read_csv(os.path.join(ROOT, "results", "raw", "model_metrics.csv"))
    groups = {}
    keys = ("model", "dataset", "prediction_type", "training_type", "resource")
    for row in rows:
        group_key = tuple(row.get(key, "") for key in keys)
        groups.setdefault(group_key, []).append(row)
    aggregated = []
    for group_key, values in groups.items():
        output = dict(zip(keys, group_key))
        output["run_count"] = len(values)
        for field in numeric_fields:
            valid = []
            for row in values:
                try:
                    number = float(row[field])
                    if math.isfinite(number):
                        valid.append(number)
                except (ValueError, KeyError, TypeError):
                    pass
            if valid:
                output[field + "_mean"] = statistics.mean(valid)
                output[field + "_median"] = statistics.median(valid)
                output[field + "_std"] = statistics.pstdev(valid)
                output[field + "_min"] = min(valid)
                output[field + "_max"] = max(valid)
        aggregated.append(output)
    out_path = os.path.join(ROOT, "results", "aggregated", "model_metrics.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as output_file:
        if aggregated:
            columns = list(dict.fromkeys(key for row in aggregated for key in row))
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(aggregated)

    service_rows = read_csv(os.path.join(ROOT, "results", "raw", "service_level_metrics.csv"))
    replace_csv(os.path.join(ROOT, "results", "metrics", "service_level_metrics.csv"), service_rows)
    replace_csv(os.path.join(ROOT, "results", "metrics", "model_metrics.csv"), rows)
    replace_csv(os.path.join(ROOT, "results", "metrics", "runtime_metrics.csv"),
                read_csv(os.path.join(ROOT, "results", "raw", "runtime_metrics.csv")))
    replace_csv(os.path.join(ROOT, "results", "metrics", "transfer_learning_metrics.csv"),
                read_csv(os.path.join(ROOT, "results", "raw", "transfer_learning_metrics.csv")))
    replace_csv(os.path.join(ROOT, "results", "metrics", "uncertainty_metrics.csv"),
                read_csv(os.path.join(ROOT, "results", "raw", "uncertainty_metrics.csv")))
    replace_csv(os.path.join(ROOT, "results", "metrics", "uncertainty_summary.csv"),
                read_csv(os.path.join(ROOT, "results", "raw", "uncertainty_summary.csv")))
    replace_csv(os.path.join(ROOT, "results", "metrics", "quantile_metrics.csv"),
                read_csv(os.path.join(ROOT, "results", "raw", "quantile_metrics.csv")))

    grouped_metrics = (
        ("quantile_metrics", ("model", "dataset", "prediction_type", "training_type", "resource", "quantile"), ("QL",)),
        ("service_level_metrics", ("model", "dataset", "prediction_type", "training_type", "resource", "service_level"),
         ("SR", "TPR", "coverage", "calibration_error", "mean_interval_width")),
        ("runtime_metrics", ("model", "dataset", "prediction_type", "training_type"),
         ("training_time", "fine_tuning_time", "inference_time", "inference_time_median",
          "inference_time_std", "number_of_parameters", "model_size_bytes", "peak_memory_mb")),
        ("transfer_learning_metrics", ("scenario", "source_dataset", "target_dataset", "resource", "model"),
         ("MSE", "MAE", "RMSE", "QL", "SR", "TPR", "training_time", "fine_tuning_time", "inference_time")),
                ("uncertainty_summary", ("model", "dataset", "prediction_type", "training_type", "resource"),
                 ("mean_prediction", "prediction_std_mean", "prediction_std_median", "prediction_std_std",
                    "epistemic_uncertainty", "aleatoric_uncertainty", "total_uncertainty")),
    )
    for filename, key_fields, metric_fields in grouped_metrics:
        source_rows = read_csv(os.path.join(ROOT, "results", "raw", filename + ".csv"))
        aggregate_rows = {}
        for source_row in source_rows:
            key = tuple(source_row.get(field, "") for field in key_fields)
            aggregate_rows.setdefault(key, []).append(source_row)
        summary_rows = []
        for key, group in aggregate_rows.items():
            summary = dict(zip(key_fields, key))
            summary["run_count"] = len(group)
            for metric in metric_fields:
                values = []
                for item in group:
                    try:
                        value = float(item[metric])
                        if math.isfinite(value):
                            values.append(value)
                    except (KeyError, ValueError, TypeError):
                        pass
                if values:
                    summary[metric + "_mean"] = statistics.mean(values)
                    summary[metric + "_median"] = statistics.median(values)
                    summary[metric + "_std"] = statistics.pstdev(values)
                    summary[metric + "_min"] = min(values)
                    summary[metric + "_max"] = max(values)
            summary_rows.append(summary)
        replace_csv(os.path.join(ROOT, "results", "aggregated", filename + ".csv"), summary_rows)

    table_rows = []
    for model_label in (
        "S-U-LSTM", "S-U-LSTMD", "S-U-HBNN", "S-B-LSTM", "S-B-LSTMD", "S-B-HBNN",
        "M-U-LSTM", "M-U-LSTMD", "M-U-HBNN", "M-B-LSTM", "M-B-LSTMD", "M-B-HBNN",
    ):
        row = {"Model": model_label}
        for resource, short in (("avgcpu", "Processing Units"), ("avgmem", "Memory")):
            selected = [item for item in aggregated
                        if item.get("model") == model_label and item.get("resource") == resource]
            if selected:
                for metric in ("MSE", "MAE"):
                    values = [float(item[metric + "_mean"]) for item in selected
                              if item.get(metric + "_mean") not in (None, "")]
                    row[short + " " + metric] = statistics.mean(values) if values else ""
        table_rows.append(row)
    table_path = os.path.join(ROOT, "results", "aggregated", "paper_model_comparison.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["Model", "Processing Units MSE", "Processing Units MAE", "Memory MSE", "Memory MAE"],
        )
        writer.writeheader()
        writer.writerows(table_rows)


def main():
    args = parse_args()
    if args.mc_samples < 1 or args.inference_repeats < 1:
        raise SystemExit("--mc-samples and --inference-repeats must be positive")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    try:
        import numpy  # noqa: F401
        import tensorflow  # noqa: F401
        import tensorflow_probability  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Missing ML dependency: {}. Install the CPU environment with: "
            "python3 -m pip install -r requirements-cpu.txt".format(error)
        )

    from util.experiment_data import available_datasets

    datasets = available_datasets()
    if not datasets:
        raise SystemExit("No processed traces found under saved_data/preprocessed")
    selected = args.datasets or (["gc19_a"] if args.mode in ("quick", "runtime") else datasets)
    unknown = [dataset for dataset in selected if dataset not in datasets]
    if unknown:
        raise SystemExit("Unknown processed dataset(s): " + ", ".join(unknown))
    if args.models:
        models = args.models
    else:
        models = ["LSTM", "HBNN", "LSTMD"]

    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    invocation_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(
        filename=os.path.join(ROOT, "results", "logs", "run-" + invocation_id + ".log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Starting %s with %s datasets, runs=%s", args.mode, len(selected), args.runs)
    save_dataset_inventory(datasets, args, invocation_id)

    if args.mode in ("quick", "runtime"):
        for model_name in models:
            run_regular_task(
                model_name, selected[0], "bivariate", "single", datasets, args, invocation_id
            )
    elif args.mode == "transfer_learning":
        run_transfer(args, selected, invocation_id)
    else:
        for model_name in models:
            for prediction_type in ("univariate", "bivariate"):
                resource_sets = (("avgcpu",), ("avgmem",)) if prediction_type == "univariate" else ((None,),)
                for resource_set in resource_sets:
                    for training_type in ("single", "multiple"):
                        target_datasets = selected if training_type == "single" else [selected[0]]
                        for dataset in target_datasets:
                            run_regular_task(
                                model_name, dataset, prediction_type, training_type,
                                datasets, args, invocation_id,
                                resource_name=resource_set[0],
                            )
        if args.mode == "full":
            run_transfer(args, selected, invocation_id)

    aggregate_metrics()
    if not args.no_plots:
        try:
            from util.experiment_plots import generate_plots
            generate_plots()
        except ImportError as error:
            logging.exception("Plot generation unavailable")
            print("Plot generation skipped because a dependency is missing: {}".format(error))
    print("Raw run data: results/raw/")
    print("Aggregated tables: results/aggregated/")
    print("Metrics: results/metrics/")
    print("Predictions: results/predictions/")
    print("Plots: results/plots/")


if __name__ == "__main__":
    main()