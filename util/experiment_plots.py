"""Generate experiment figures exclusively from saved CSV outputs."""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")


def read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def save_figure(name):
    os.makedirs(PLOTS, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, name), dpi=150)
    plt.close()


def plot_prediction_files():
    for path in sorted(glob.glob(os.path.join(RESULTS, "predictions", "*.csv"))):
        rows = read_rows(path)
        resources = sorted(set(row.get("resource", "") for row in rows))
        if not rows or not resources:
            continue
        figure, axes = plt.subplots(len(resources), 1, figsize=(12, 4 * len(resources)), squeeze=False)
        for axis, resource in zip(axes[:, 0], resources):
            selected = [row for row in rows if row.get("resource") == resource][:500]
            actual = [number(row, "actual") for row in selected]
            mean = [number(row, "prediction_mean") for row in selected]
            lower = [number(row, "lower_bound_95") for row in selected]
            upper = [number(row, "upper_bound_95") for row in selected]
            x_values = list(range(len(selected)))
            axis.plot(x_values, actual, label="Actual", linewidth=1.1)
            axis.plot(x_values, mean, label="Prediction mean", linewidth=1.1)
            if all(value is not None for value in lower + upper):
                axis.fill_between(x_values, lower, upper, alpha=0.2, label="95% interval")
            axis.set_title("{}: {}".format(os.path.basename(path), resource))
            axis.set_xlabel("Test sample")
            axis.set_ylabel("Normalized workload")
            axis.legend()
            axis.grid(alpha=0.25)
        save_figure("prediction_" + os.path.basename(path).replace(".csv", ".png"))


def plot_point_metrics(rows):
    for metric in ("MSE", "MAE", "RMSE"):
        grouped = {}
        for row in rows:
            value = number(row, metric)
            if value is None:
                continue
            key = (row.get("model", ""), row.get("resource", ""))
            grouped.setdefault(key, []).append(value)
        if not grouped:
            continue
        labels = ["{}\n{}".format(model, resource) for model, resource in grouped]
        values = [sum(grouped[key]) / len(grouped[key]) for key in grouped]
        plt.figure(figsize=(max(9, len(labels) * 0.8), 5))
        plt.bar(range(len(labels)), values, color="#287c8e")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title("{} comparison".format(metric))
        plt.grid(axis="y", alpha=0.25)
        save_figure("{}_comparison.png".format(metric.lower()))


def plot_quantiles(rows):
    if not rows:
        return
    grouped = {}
    for row in rows:
        quantile = number(row, "quantile")
        loss = number(row, "QL")
        if quantile is not None and loss is not None:
            grouped.setdefault((row.get("model", ""), row.get("resource", "")), []).append(
                (quantile, loss)
            )
    if not grouped:
        return
    plt.figure(figsize=(10, 6))
    for (model, resource), points in grouped.items():
        points.sort()
        plt.plot([item[0] * 100 for item in points], [item[1] for item in points],
                 marker=".", label="{} {}".format(model, resource))
    plt.xlabel("Quantile (%)")
    plt.ylabel("Mean pinball loss")
    plt.title("Quantile loss comparison")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save_figure("quantile_loss_comparison.png")


def plot_service_metrics(rows):
    if not rows:
        return
    grouped = {}
    for row in rows:
        level = number(row, "service_level")
        sr = number(row, "SR")
        tpr = number(row, "TPR")
        if level is None or sr is None or tpr is None:
            continue
        grouped.setdefault((row.get("model", ""), row.get("resource", "")), []).append(
            (level * 100.0, sr, tpr)
        )
    if not grouped:
        return

    plt.figure(figsize=(9, 6))
    for key, points in grouped.items():
        points.sort()
        plt.plot([p[0] for p in points], [p[1] for p in points], marker=".", label="{} {}".format(*key))
    plt.plot([90, 99.5], [90, 99.5], "k--", linewidth=1, label="Ideal calibration")
    plt.xlabel("Target service level (%)")
    plt.ylabel("Success rate (%)")
    plt.title("Service level vs success rate")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save_figure("service_level_vs_success_rate.png")

    plt.figure(figsize=(9, 6))
    for key, points in grouped.items():
        points.sort()
        plt.plot([p[0] for p in points], [p[2] for p in points], marker=".", label="{} {}".format(*key))
    plt.xlabel("Target service level (%)")
    plt.ylabel("Total predicted resources (% of actual demand)")
    plt.title("Service level vs TPR")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save_figure("service_level_vs_tpr.png")

    plt.figure(figsize=(9, 6))
    for key, points in grouped.items():
        points.sort(key=lambda item: item[2])
        plt.plot([p[2] for p in points], [p[1] for p in points], marker=".", label="{} {}".format(*key))
    plt.xlabel("Total predicted resources (% of actual demand)")
    plt.ylabel("Success rate (%)")
    plt.title("TPR vs SR (paper-style service-level trade-off)")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    save_figure("tpr_vs_sr.png")


def plot_runtime(rows):
    for field, title, filename in (
        ("training_time", "Training time", "training_time_comparison.png"),
        ("fine_tuning_time", "Fine-tuning time", "fine_tuning_time_comparison.png"),
        ("inference_time", "Single-sample inference latency", "inference_time_comparison.png"),
    ):
        grouped = {}
        for row in rows:
            value = number(row, field)
            if value is not None:
                grouped.setdefault(row.get("model", ""), []).append(value)
        if not grouped:
            continue
        labels = list(grouped)
        values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
        plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
        plt.bar(range(len(labels)), values, color="#e07a5f")
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
        plt.ylabel("Seconds")
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        save_figure(filename)


def plot_uncertainty():
    paths = sorted(glob.glob(os.path.join(RESULTS, "predictions", "*.csv")))
    figure_created = False
    plt.figure(figsize=(10, 5))
    for path in paths:
        rows = read_rows(path)
        selected = [row for row in rows if number(row, "epistemic_uncertainty") is not None
                    and number(row, "aleatoric_uncertainty") is not None]
        if not selected:
            continue
        selected = selected[:500]
        plt.plot([number(row, "epistemic_uncertainty") for row in selected],
                 label=os.path.basename(path) + " epistemic", alpha=0.8)
        plt.plot([number(row, "aleatoric_uncertainty") for row in selected],
                 label=os.path.basename(path) + " aleatoric", alpha=0.8, linestyle="--")
        figure_created = True
    if figure_created:
        plt.xlabel("Test sample")
        plt.ylabel("Predicted standard deviation")
        plt.title("Epistemic and aleatoric uncertainty")
        plt.legend(fontsize=6)
        plt.grid(alpha=0.25)
        save_figure("epistemic_vs_aleatoric_uncertainty.png")
    else:
        plt.close()


def plot_transfer(rows):
    if not rows:
        return
    groups = {}
    for row in rows:
        value = number(row, "MSE")
        if value is not None:
            key = (row.get("scenario", ""), row.get("target_group", ""))
            groups.setdefault(key, []).append(value)
    if not groups:
        return
    labels = ["{}\n{}".format(*key) for key in groups]
    values = [sum(groups[key]) / len(groups[key]) for key in groups]
    plt.figure(figsize=(max(10, len(labels) * 0.7), 5))
    plt.bar(range(len(labels)), values, color="#6a994e")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("MSE")
    plt.title("Transfer learning performance by scenario")
    plt.grid(axis="y", alpha=0.25)
    save_figure("transfer_learning_performance.png")

    fine_tune_groups = {"zero-shot": [], "fine-tuned": []}
    for row in rows:
        value = number(row, "MSE")
        if value is not None:
            label = "fine-tuned" if row.get("fine_tuning", "").lower() == "true" else "zero-shot"
            fine_tune_groups[label].append(value)
    if all(fine_tune_groups.values()):
        plt.figure(figsize=(7, 5))
        plt.boxplot([fine_tune_groups["zero-shot"], fine_tune_groups["fine-tuned"]],
                    labels=["Zero-shot", "Fine-tuned"])
        plt.ylabel("MSE")
        plt.title("Zero-shot vs fine-tuned performance")
        plt.grid(axis="y", alpha=0.25)
        save_figure("zero_shot_vs_fine_tuned.png")


def plot_group_comparisons(rows):
    for grouping, filename, title in (
        (lambda name: name.split("-")[1] if "-" in name else "unknown",
         "univariate_vs_bivariate.png", "Univariate vs bivariate performance"),
        (lambda name: name.split("-")[0] if "-" in name else "unknown",
         "single_vs_multiple_dataset.png", "Single vs multiple-dataset performance"),
    ):
        grouped = {}
        for row in rows:
            value = number(row, "MSE")
            if value is not None:
                grouped.setdefault(grouping(row.get("model", "")), []).append(value)
        if len(grouped) < 2:
            continue
        labels = list(grouped)
        values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
        plt.figure(figsize=(7, 5))
        plt.bar(labels, values, color="#bc6c25")
        plt.ylabel("MSE")
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        save_figure(filename)

    by_dataset = {}
    for row in rows:
        value = number(row, "MSE")
        dataset = row.get("dataset", "")
        if value is not None and dataset:
            by_dataset.setdefault(dataset, []).append(value)
    if by_dataset:
        labels = list(by_dataset)
        values = [sum(by_dataset[label]) / len(by_dataset[label]) for label in labels]
        plt.figure(figsize=(max(9, len(labels) * 0.7), 5))
        plt.bar(range(len(labels)), values, color="#52796f")
        plt.xticks(range(len(labels)), labels, rotation=40, ha="right")
        plt.ylabel("MSE")
        plt.title("Dataset-wise performance")
        plt.grid(axis="y", alpha=0.25)
        save_figure("dataset_wise_performance.png")


def generate_plots():
    os.makedirs(PLOTS, exist_ok=True)
    model_rows = read_rows(os.path.join(RESULTS, "raw", "model_metrics.csv"))
    service_rows = read_rows(os.path.join(RESULTS, "raw", "service_level_metrics.csv"))
    quantile_rows = read_rows(os.path.join(RESULTS, "raw", "quantile_metrics.csv"))
    runtime_rows = read_rows(os.path.join(RESULTS, "raw", "runtime_metrics.csv"))
    transfer_rows = read_rows(os.path.join(RESULTS, "raw", "transfer_learning_metrics.csv"))
    plot_prediction_files()
    plot_point_metrics(model_rows)
    plot_quantiles(quantile_rows)
    plot_service_metrics(service_rows)
    plot_runtime(runtime_rows)
    plot_uncertainty()
    plot_transfer(transfer_rows)
    plot_group_comparisons(model_rows)
