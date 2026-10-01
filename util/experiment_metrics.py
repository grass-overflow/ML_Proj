"""Metric definitions used by the workload reproduction experiment runner."""

import math
from statistics import NormalDist

import numpy as np


DEFAULT_SERVICE_LEVELS = tuple(
    [level / 100.0 for level in range(90, 100)] + [0.995]
)
DEFAULT_QUANTILES = tuple(sorted(set([0.5] + list(DEFAULT_SERVICE_LEVELS))))


def _as_2d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("Expected a vector or a 2D array of resource values")
    return values


def point_metrics(actual, prediction_mean):
    """Compute point scores from actuals and the model's point estimate."""
    actual = _as_2d(actual)
    prediction_mean = _as_2d(prediction_mean)
    if actual.shape != prediction_mean.shape:
        raise ValueError("Actual and predicted arrays must have equal shapes")
    errors = prediction_mean - actual
    mse = float(np.mean(np.square(errors)))
    variance = float(np.sum(np.square(actual - np.mean(actual))))
    r_squared = (
        float(1.0 - np.sum(np.square(errors)) / variance)
        if variance > 0.0
        else float("nan")
    )
    return {
        "MSE": mse,
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": math.sqrt(mse),
        "R2": r_squared,
    }


def gaussian_quantile(mean, standard_deviation, quantile):
    """Return elementwise Gaussian quantiles for an explicit probability."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    mean = _as_2d(mean)
    standard_deviation = np.maximum(_as_2d(standard_deviation), 0.0)
    if mean.shape != standard_deviation.shape:
        raise ValueError("Mean and standard deviation arrays must have equal shapes")
    z_score = NormalDist().inv_cdf(quantile)
    return mean + z_score * standard_deviation


def pinball_loss(actual, predicted_quantile, quantile):
    """Mean pinball loss, averaged across timestamps and resources."""
    actual = _as_2d(actual)
    predicted_quantile = _as_2d(predicted_quantile)
    if actual.shape != predicted_quantile.shape:
        raise ValueError("Actual and quantile prediction arrays must match")
    error = actual - predicted_quantile
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def probabilistic_metrics(
    actual,
    prediction_mean,
    prediction_std,
    service_levels=DEFAULT_SERVICE_LEVELS,
    quantile_levels=DEFAULT_QUANTILES,
):
    """Compute marginal Gaussian quantile, interval, and service-level metrics.

    SR is 100 * P(actual <= one-sided service-level quantile), matching the
    paper's success-rate definition. TPR is 100 * sum(upper bounds) /
    sum(actual demand). Central interval coverage is reported separately and
    uses equal tail probabilities; it is not the paper's one-sided SR.
    """
    actual = _as_2d(actual)
    prediction_mean = _as_2d(prediction_mean)
    prediction_std = np.maximum(_as_2d(prediction_std), 0.0)
    if actual.shape != prediction_mean.shape or actual.shape != prediction_std.shape:
        raise ValueError("Actual, mean, and standard deviation arrays must match")

    quantile_rows = []
    quantile_predictions = {}
    for quantile in quantile_levels:
        predicted = gaussian_quantile(prediction_mean, prediction_std, quantile)
        quantile_predictions[quantile] = predicted
        quantile_rows.append(
            {
                "quantile": float(quantile),
                "QL": pinball_loss(actual, predicted, quantile),
            }
        )

    service_rows = []
    for service_level in service_levels:
        upper = quantile_predictions.get(service_level)
        if upper is None:
            upper = gaussian_quantile(prediction_mean, prediction_std, service_level)
        lower_probability = (1.0 - service_level) / 2.0
        lower = gaussian_quantile(prediction_mean, prediction_std, lower_probability)
        interval_upper = gaussian_quantile(
            prediction_mean, prediction_std, 1.0 - lower_probability
        )
        demand_sum = float(np.sum(actual))
        tpr = float(100.0 * np.sum(upper) / demand_sum) if demand_sum > 0.0 else float("nan")
        coverage = float(np.mean((actual >= lower) & (actual <= interval_upper)) * 100.0)
        success_rate = float(np.mean(actual <= upper) * 100.0)
        service_rows.append(
            {
                "service_level": float(service_level),
                "quantile": float(service_level),
                "SR": success_rate,
                "TPR": tpr,
                "coverage": coverage,
                "calibration_error": success_rate - service_level * 100.0,
                "mean_interval_width": float(np.mean(interval_upper - lower)),
            }
        )

    return quantile_rows, service_rows


def residual_quantile_bounds(actual_validation, prediction_validation, actual_test,
                             prediction_test, service_levels=DEFAULT_SERVICE_LEVELS):
    """Build comparison-only LSTM intervals from validation residuals.

    The test set is never used to calibrate the bounds. These bounds are not
    learned model uncertainty and must not be described as epistemic or
    aleatoric uncertainty.
    """
    actual_validation = _as_2d(actual_validation)
    prediction_validation = _as_2d(prediction_validation)
    actual_test = _as_2d(actual_test)
    prediction_test = _as_2d(prediction_test)
    if actual_validation.shape != prediction_validation.shape:
        raise ValueError("Validation actuals and predictions must match")
    if actual_test.shape != prediction_test.shape:
        raise ValueError("Test actuals and predictions must match")
    residuals = actual_validation - prediction_validation
    bounds = {}
    for service_level in service_levels:
        alpha = 1.0 - service_level
        lower_residual = np.quantile(residuals, alpha / 2.0, axis=0, keepdims=True)
        upper_residual = np.quantile(residuals, 1.0 - alpha / 2.0, axis=0, keepdims=True)
        bounds[service_level] = (
            prediction_test + lower_residual,
            prediction_test + upper_residual,
        )
    return bounds


def summarize_uncertainty(prediction_std, epistemic_std=None, aleatoric_std=None):
    """Summarize predicted marginal standard deviations by resource channel."""
    prediction_std = _as_2d(prediction_std)
    summary = {
        "prediction_std_mean": np.mean(prediction_std, axis=0),
        "prediction_std_median": np.median(prediction_std, axis=0),
        "prediction_std_std": np.std(prediction_std, axis=0),
    }
    for name, values in (
        ("epistemic_uncertainty", epistemic_std),
        ("aleatoric_uncertainty", aleatoric_std),
    ):
        summary[name] = (
            np.mean(_as_2d(values), axis=0)
            if values is not None
            else np.full(prediction_std.shape[1], np.nan)
        )
    summary["total_uncertainty"] = np.mean(prediction_std, axis=0)
    return summary