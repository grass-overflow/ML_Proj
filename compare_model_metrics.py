"""Compare exported S-B metrics from the three model-specific Kaggle runs."""

import io
import os
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


MODELS = ("LSTM", "LSTMD", "HBNN")
DATASETS = (
    "ali18", "ali20_c", "ali20_g", "gc11", "gc19_a", "gc19_b",
    "gc19_c", "gc19_d", "gc19_e", "gc19_f", "gc19_g", "gc19_h",
)
PAPER_S_B = {
    "LSTM": {"avgcpu": (0.0047, 0.0500), "avgmem": (0.0054, 0.0487)},
    "LSTMD": {"avgcpu": (0.0061, 0.0556), "avgmem": (0.0088, 0.0631)},
    "HBNN": {"avgcpu": (0.0299, 0.1285), "avgmem": (0.0386, 0.1533)},
}
PAPER_TABLE1 = {
    "S-U-LSTM": {"avgcpu": (0.0041, 0.0457), "avgmem": (0.0042, 0.0425)},
    "S-U-LSTMD": {"avgcpu": (0.0040, 0.0455), "avgmem": (0.0041, 0.0420)},
    "S-U-HBNN": {"avgcpu": (0.0047, 0.0502), "avgmem": (0.0044, 0.0455)},
    "S-B-LSTM": {"avgcpu": (0.0047, 0.0500), "avgmem": (0.0054, 0.0487)},
    "S-B-LSTMD": {"avgcpu": (0.0061, 0.0556), "avgmem": (0.0088, 0.0631)},
    "S-B-HBNN": {"avgcpu": (0.0299, 0.1285), "avgmem": (0.0386, 0.1533)},
    "M-U-LSTM": {"avgcpu": (0.0044, 0.0474), "avgmem": (0.0048, 0.0447)},
    "M-U-LSTMD": {"avgcpu": (0.0041, 0.0464), "avgmem": (0.0044, 0.0439)},
    "M-U-HBNN": {"avgcpu": (0.0047, 0.0485), "avgmem": (0.0052, 0.0498)},
    "M-B-LSTM": {"avgcpu": (0.0044, 0.0479), "avgmem": (0.0044, 0.0438)},
    "M-B-LSTMD": {"avgcpu": (0.0046, 0.0446), "avgmem": (0.0046, 0.0446)},
    "M-B-HBNN": {"avgcpu": (0.0042, 0.0471), "avgmem": (0.0043, 0.0436)},
}
COLORS = {"LSTM": "#287c8e", "LSTMD": "#e07a5f", "HBNN": "#6a994e"}
INPUT_DIR = Path(os.environ.get("MODEL_METRICS_INPUT", "/kaggle/input/model-comparison-results"))
OUTPUT_DIR = Path("results/model_comparison")


def find_unique_export(filename):
    """Find one exact named CSV, either directly or inside a model results ZIP."""
    files = [path for path in INPUT_DIR.rglob(filename) if path.is_file()]
    archived = []
    for archive_path in INPUT_DIR.rglob("*.zip"):
        if not archive_path.is_file():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            archived.extend(
                (archive_path, member)
                for member in archive.namelist()
                if Path(member).name == filename
            )

    matches = [("file", path) for path in files]
    matches.extend(("zip", item) for item in archived)
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {INPUT_DIR}. Add the model-result "
            "ZIPs (or their CSVs) to one Kaggle input dataset and set INPUT_DIR "
            "to that dataset's path."
        )
    if len(matches) > 1:
        locations = [
            str(item[1]) if item[0] == "file"
            else f"{item[1][0]}:{item[1][1]}"
            for item in matches
        ]
        raise RuntimeError(
            f"Found multiple copies of {filename}; keep only the intended run: "
            + ", ".join(locations)
        )

    kind, source = matches[0]
    if kind == "file":
        return pd.read_csv(source)
    archive_path, member = source
    with zipfile.ZipFile(archive_path) as archive:
        return pd.read_csv(io.BytesIO(archive.read(member)))


def load_and_validate_exports():
    point_frames = []
    service_frames = []
    for model in MODELS:
        point = find_unique_export(f"{model}_model_metrics.csv")
        service = find_unique_export(f"{model}_service_level_metrics.csv")

        required_point = {"model", "dataset", "resource", "MSE", "MAE"}
        required_service = {
            "model", "dataset", "resource", "service_level", "SR", "TPR",
        }
        if missing := required_point.difference(point.columns):
            raise ValueError(f"{model} point metrics are missing columns: {sorted(missing)}")
        if missing := required_service.difference(service.columns):
            raise ValueError(f"{model} service metrics are missing columns: {sorted(missing)}")

        expected_label = f"S-B-{model}"
        point = point[point["model"] == expected_label].copy()
        service = service[service["model"] == expected_label].copy()
        expected_pairs = {(dataset, resource) for dataset in DATASETS for resource in ("avgcpu", "avgmem")}
        actual_pairs = set(zip(point["dataset"], point["resource"]))
        if actual_pairs != expected_pairs or len(point) != len(expected_pairs):
            raise ValueError(
                f"{model} must have exactly one CPU and memory point-metric row "
                f"for each of the 12 traces; found {len(point)} rows."
            )
        service_pairs = set(zip(service["dataset"], service["resource"]))
        if service_pairs != expected_pairs:
            raise ValueError(
                f"{model} service metrics do not cover both resources for all 12 traces."
            )
        expected_service_rows = len(expected_pairs) * 11
        if len(service) != expected_service_rows or service["service_level"].nunique() != 11:
            raise ValueError(
                f"{model} must have 11 service levels for each trace/resource "
                f"({expected_service_rows} rows); found {len(service)}."
            )
        point_frames.append(point)
        service_frames.append(service)

    return pd.concat(point_frames, ignore_index=True), pd.concat(service_frames, ignore_index=True)


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR}. Set MODEL_METRICS_INPUT "
            "to the mounted Kaggle dataset directory."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    point, service = load_and_validate_exports()
    point.to_csv(OUTPUT_DIR / "all_models_model_metrics.csv", index=False)
    service.to_csv(OUTPUT_DIR / "all_models_service_level_metrics.csv", index=False)

    paper_rows = []
    for label, resources in PAPER_TABLE1.items():
        for resource, (paper_mse, paper_mae) in resources.items():
            selected = point[
                (point["model"] == label) & (point["resource"] == resource)
            ]
            paper_rows.append({
                "model": label,
                "resource": resource,
                "paper_MSE": paper_mse,
                "run_MSE_mean_across_12_traces": selected["MSE"].mean() if len(selected) else None,
                "paper_MAE": paper_mae,
                "run_MAE_mean_across_12_traces": selected["MAE"].mean() if len(selected) else None,
                "traces_in_run": len(selected),
            })
    paper_table = pd.DataFrame(paper_rows)
    paper_table.to_csv(OUTPUT_DIR / "paper_table1_comparison.csv", index=False)
    display_table = paper_table[paper_table["model"].str.startswith("S-B-")].copy()
    display(display_table)
    print(
        "Only S-B (single-training, bivariate) was trained here. S-U, M-U, and "
        "M-B paper rows are shown as references with no run result; they require "
        "additional training configurations."
    )
    print(
        "Run-vs-paper values are descriptive unweighted means over 12 traces "
        "(one seed), not a direct replication of the paper's aggregate."
    )

    sb = point.copy()
    sb["architecture"] = sb["model"].str.removeprefix("S-B-")
    summary = (
        sb.groupby(["architecture", "resource"], as_index=False)
        .agg(MSE=("MSE", "mean"), MAE=("MAE", "mean"), trace_count=("dataset", "nunique"))
    )
    summary.to_csv(OUTPUT_DIR / "mean_metrics_by_model_resource.csv", index=False)

    # The paper Table 1 comparison, restricted to the S-B rows actually run.
    plotted = display_table.reset_index(drop=True)
    categories = [f"{row.model}\n{row.resource}" for row in plotted.itertuples()]
    x = range(len(plotted))
    for metric, paper_column, run_column, ylabel in (
        ("MSE", "paper_MSE", "run_MSE_mean_across_12_traces", "MSE"),
        ("MAE", "paper_MAE", "run_MAE_mean_across_12_traces", "MAE"),
    ):
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.bar([value - 0.2 for value in x], plotted[paper_column], width=0.4, label="Paper Table 1")
        axis.bar([value + 0.2 for value in x], plotted[run_column], width=0.4, label="This run: mean across 12 traces")
        axis.set_xticks(list(x), categories)
        axis.set_ylabel(ylabel)
        axis.set_title(f"S-B {metric}: paper reference vs three-model run")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"paper_vs_run_{metric.lower()}.png", dpi=150)
        display(fig)
        plt.close(fig)

    # Per-trace point scores, with one panel for each resource.
    for metric in ("MSE", "MAE"):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        for axis, resource in zip(axes, ("avgcpu", "avgmem")):
            subset = sb[sb["resource"] == resource]
            for model in MODELS:
                label = f"S-B-{model}"
                values = subset[subset["model"] == label].set_index("dataset")[metric].reindex(DATASETS)
                axis.plot(DATASETS, values, marker="o", color=COLORS[model], label=model)
            axis.set_title(f"{resource}: {metric} by trace")
            axis.set_ylabel(metric)
            axis.legend()
            axis.grid(alpha=0.25)
        axes[-1].tick_params(axis="x", rotation=40)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{metric.lower()}_by_trace.png", dpi=150)
        display(fig)
        plt.close(fig)

    service["architecture"] = service["model"].str.removeprefix("S-B-")
    mean_service = (
        service.groupby(["architecture", "resource", "service_level"], as_index=False)
        .agg(SR=("SR", "mean"), TPR=("TPR", "mean"), trace_count=("dataset", "nunique"))
    )
    mean_service.to_csv(OUTPUT_DIR / "mean_service_metrics_by_model_resource.csv", index=False)

    # Each point is the unweighted mean of the 12 per-trace service metrics.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, resource in zip(axes, ("avgcpu", "avgmem")):
        subset = mean_service[mean_service["resource"] == resource]
        for model in MODELS:
            curve = subset[subset["architecture"] == model].sort_values("service_level")
            axis.plot(
                curve["service_level"] * 100,
                curve["SR"],
                marker="o",
                color=COLORS[model],
                label=model,
            )
        axis.plot([90, 99.5], [90, 99.5], "k--", linewidth=1, label="Ideal calibration")
        axis.set_title(f"{resource}: confidence vs. success rate")
        axis.set_xlabel("Target confidence / service level (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Success rate, SR (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confidence_vs_sr.png", dpi=150)
    display(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, resource in zip(axes, ("avgcpu", "avgmem")):
        subset = mean_service[mean_service["resource"] == resource]
        for model in MODELS:
            curve = subset[subset["architecture"] == model].sort_values("TPR")
            axis.plot(
                curve["TPR"],
                curve["SR"],
                marker="o",
                color=COLORS[model],
                label=model,
            )
        axis.set_title(f"{resource}: TPR vs. SR")
        axis.set_xlabel("Total predicted resources / actual demand, TPR (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Success rate, SR (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tpr_vs_sr.png", dpi=150)
    display(fig)
    plt.close(fig)

    archive_path = Path(shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR))
    print(f"Comparison tables, CSVs, and plots saved in: {OUTPUT_DIR}")
    print(f"Downloadable archive: {archive_path}")
    try:
        from IPython.display import FileLink
        display(FileLink(str(archive_path)))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
