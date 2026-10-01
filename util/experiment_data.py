"""Low-memory, chronological data preparation for the experiment runner."""

import csv
import math
import os

import numpy as np


DATA_DIRECTORY = os.path.join("saved_data", "preprocessed")
RESOURCE_COLUMNS = ("avgcpu", "avgmem")
SAMPLE_INTERVAL_SECONDS = 300


def available_datasets(data_directory=DATA_DIRECTORY):
    """Return processed trace names, in deterministic order."""
    if not os.path.isdir(data_directory):
        return []
    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(data_directory)
        if filename.lower().endswith(".csv")
    )


def _read_trace(dataset, resources, data_directory=DATA_DIRECTORY):
    path = os.path.join(data_directory, dataset + ".csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("Processed workload trace not found: " + path)

    with open(path, "r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing = [resource for resource in resources if resource not in columns]
        if missing:
            raise ValueError(
                "Trace {} is missing required columns: {}".format(
                    dataset, ", ".join(missing)
                )
            )
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("Trace {} has fewer than two samples".format(dataset))

    values = np.asarray(
        [[float(row[resource]) for resource in resources] for row in rows],
        dtype=np.float32,
    )
    timestamps = np.asarray(
        [row.get("time", "") for row in rows], dtype=object
    )
    if not np.isfinite(values).all():
        raise ValueError("Trace {} contains NaN or infinite values".format(dataset))
    return values, timestamps


def load_dataset_split(
    dataset,
    resources,
    window_size=288,
    horizon_steps=2,
    train_fraction=0.8,
    validation_fraction=0.2,
    max_train_windows=None,
    max_test_windows=None,
    data_directory=DATA_DIRECTORY,
):
    """Build windows and split strictly by target timestamp.

    The horizon is measured from the final observed sample. Thus a horizon of
    two with five-minute samples predicts ten minutes after the input window.
    Test windows may use prior history as input, but no test target enters
    training or validation.
    """
    if not resources:
        raise ValueError("At least one resource column is required")
    if window_size < 1 or horizon_steps < 1:
        raise ValueError("window_size and horizon_steps must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    values, timestamps = _read_trace(dataset, resources, data_directory)
    sample_count = values.shape[0]
    boundary = int(math.floor(sample_count * train_fraction))
    final_start = sample_count - window_size - horizon_steps + 1
    if final_start <= 0:
        raise ValueError(
            "Trace {} is too short for window {} and horizon {}".format(
                dataset, window_size, horizon_steps
            )
        )

    # sliding_window_view returns a view until the selected split is copied.
    windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(
            values, window_shape=window_size, axis=0
        ),
        -1,
        1,
    )[:final_start]
    starts = np.arange(final_start, dtype=np.int64)
    target_indices = starts + window_size - 1 + horizon_steps
    train_indices = np.flatnonzero(target_indices < boundary)
    test_indices = np.flatnonzero(target_indices >= boundary)

    if max_train_windows is not None and len(train_indices) > max_train_windows:
        stride = int(math.ceil(len(train_indices) / float(max_train_windows)))
        train_indices = train_indices[::stride]
    if max_test_windows is not None and len(test_indices) > max_test_windows:
        stride = int(math.ceil(len(test_indices) / float(max_test_windows)))
        test_indices = test_indices[::stride]

    if len(train_indices) < 2 or len(test_indices) < 1:
        raise ValueError(
            "Trace {} does not provide enough train/test windows".format(dataset)
        )

    train_count = len(train_indices)
    validation_count = int(math.floor(train_count * validation_fraction))
    fit_count = train_count - validation_count
    train_indices = train_indices[:fit_count]
    validation_indices = np.flatnonzero(target_indices < boundary)[fit_count:]

    if max_train_windows is not None:
        # Keep the bounded quick subset chronological while retaining validation.
        all_train_indices = np.flatnonzero(target_indices < boundary)
        if len(all_train_indices) > max_train_windows:
            stride = int(math.ceil(len(all_train_indices) / float(max_train_windows)))
            all_train_indices = all_train_indices[::stride]
        validation_count = int(math.floor(len(all_train_indices) * validation_fraction))
        fit_count = len(all_train_indices) - validation_count
        train_indices = all_train_indices[:fit_count]
        validation_indices = all_train_indices[fit_count:]

    result = {
        "dataset": dataset,
        "resources": list(resources),
        "X_train": np.ascontiguousarray(windows[train_indices], dtype=np.float32),
        "y_train": np.ascontiguousarray(values[target_indices[train_indices]], dtype=np.float32),
        "X_val": np.ascontiguousarray(windows[validation_indices], dtype=np.float32),
        "y_val": np.ascontiguousarray(values[target_indices[validation_indices]], dtype=np.float32),
        "X_test": np.ascontiguousarray(windows[test_indices], dtype=np.float32),
        "y_test": np.ascontiguousarray(values[target_indices[test_indices]], dtype=np.float32),
        "test_timestamps": timestamps[target_indices[test_indices]],
        "train_target_indices": target_indices[train_indices],
        "validation_target_indices": target_indices[validation_indices],
        "test_target_indices": target_indices[test_indices],
        "target_indices": target_indices,
        "train_boundary": boundary,
        "sample_count": sample_count,
        "feature_count": values.shape[1],
        "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(test_indices),
    }
    return result


def combine_training_splits(splits):
    """Combine source-domain train/validation arrays in bounded batches."""
    if not splits:
        raise ValueError("At least one source dataset is required")
    return {
        "X_train": np.concatenate([split["X_train"] for split in splits], axis=0),
        "y_train": np.concatenate([split["y_train"] for split in splits], axis=0),
        "X_val": np.concatenate([split["X_val"] for split in splits], axis=0),
        "y_val": np.concatenate([split["y_val"] for split in splits], axis=0),
    }