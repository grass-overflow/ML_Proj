# Experimental Methodology

## Paper Setup And Local Setup

The reference paper uses 288 five-minute observations (24 hours) to forecast demand two five-minute steps beyond the final input observation. It trains on the first 80% of each trace, reserves 20% of the training portion for validation, and evaluates the final 20%. Its reported runtime hardware was an Intel Xeon Gold 6240 at 2.60 GHz and an NVIDIA Quadro RTX 8000 with 48 GB on Ubuntu 20.04.

This checkout targets the user's 8 GB, CPU-only machine and Python 3.9-3.12. TensorFlow CPU 2.16.2 and TensorFlow Probability 0.24.0 are pinned in `requirements-cpu.txt`; the paper used an older GPU stack. Quick mode caps fit windows at 512, test windows at 256, epochs at 3, and batch size at 64. Standard model-comparison runs cap epochs at 100 and batch size at 128 unless overridden. These are resource controls, not reproduction results. The default number of seeds is three.

## Data And Splitting

The runner uses the preprocessed CSVs in `saved_data/preprocessed/`; no raw-trace download or second normalization pass occurs. All 12 available traces use five-minute samples and the columns `avgcpu` and `avgmem`. Google 2011 and 2019 and Alibaba 2018 and 2020 labels are inferred from the processed filenames. An inventory is saved to `results/metrics/dataset_summary.csv`.

For a window beginning at row $i$, with window size $w$ and horizon $h$, inputs are rows $i$ through $i+w-1$ and the target is row $i+w-1+h$. At the default $w=288$, $h=2$, and five-minute sampling, the target is ten minutes after the last input observation. A window is assigned to train/validation or test based on its target row, not its starting row. The first 80% of target rows form the train/validation segment; the remaining 20% form the test segment. The last 20% of the first segment is validation, so about 64%/16%/20% of all eligible targets are used for fitting/validation/testing. Test windows can look back across the split boundary as normal forecasting context, but test targets never enter training, validation, hyperparameter selection, or LSTM residual calibration.

Multi-dataset training concatenates source training windows and validation windows without random shuffling. The implementation uses float32 windows and can downsample chronologically in quick mode. It does not retain raw traces in duplicate after building each split, although the combined training arrays must coexist during model fitting.

## Model Configurations

The runner retains the project implementations in `models/LSTM.py`, `models/HBNN.py`, and `models/LSTMD.py`:

- LSTM: Conv1D + LSTM + dense point output, trained with MSE.
- HBNN: Conv1D + LSTM + dense layers + Bayesian DenseVariational layer + independent Gaussian output. It models epistemic and aleatoric uncertainty.
- LSTMD: Conv1D + LSTM + dense layers + independent Gaussian output. It models aleatoric uncertainty only.

Univariate models predict either CPU or memory. Bivariate models predict CPU and memory jointly. The naming is `S-U`, `S-B`, `M-U`, and `M-B` for single/multiple training and univariate/bivariate prediction; appending the architecture gives the paper labels (for example `M-B-HBNN`). Parameter rows come from the existing Talos CSVs when a matching file is available. Talos search is not run by the normal pipeline. Quick uses `gc19_a` with S-B for the three model families. `model_comparison` covers all four configuration types; multiple-dataset cases use all processed traces. `full` adds transfer scenarios.

Saved hyperparameters are retained but capped for practical runs: quick uses no more than three epochs, patience two, and batch size 64; other modes use no more than 100 epochs, patience eight, and batch size 128 by default. Command-line overrides are recorded in result metadata where applicable.

## Uncertainty And Metrics

For HBNN, multiple stochastic Bayesian forward passes produce means $\mu_s$ and aleatoric variances $\sigma_s^2$. The reported values are:

$$
\hat{\mu} = \frac{1}{S}\sum_s \mu_s,\qquad
\sigma^2_{\mathrm{epistemic}} = \mathrm{Var}_s(\mu_s),\qquad
\sigma^2_{\mathrm{aleatoric}} = \frac{1}{S}\sum_s \sigma_s^2,\qquad
\sigma^2_{\mathrm{total}} = \sigma^2_{\mathrm{epistemic}} + \sigma^2_{\mathrm{aleatoric}}.
$$

LSTMD's predicted Gaussian standard deviation is aleatoric uncertainty; it has no epistemic estimate. LSTM's deterministic output has no learned uncertainty. For baseline comparison only, LSTM residual intervals are calibrated from validation residuals and are explicitly not treated as model uncertainty.

MSE, MAE, RMSE, and R² use actual targets and the point prediction (the distribution mean for HBNN/LSTMD). Quantile loss is the mean pinball loss:

$$
\ell_q(y,\hat y_q) = \max(q(y-\hat y_q), (q-1)(y-\hat y_q)).
$$

The supported quantiles include 0.50, 0.90 through 0.99, and 0.995. Probabilistic-model quantiles assume the marginal Gaussian distribution returned by each resource output.

For a requested one-sided service level $q$, the service upper bound is the model's $q$ quantile. SR follows the paper's upper-bound definition:

$$
\mathrm{SR}(q) = 100\cdot\frac{\#\{t:y_t\leq U_{t,q}\}}{N},\qquad
\mathrm{TPR}(q) = 100\cdot\frac{\sum_t U_{t,q}}{\sum_t y_t}.
$$

TPR is calculated on the same target scale as the supplied processed trace; it is a percentage of aggregate observed demand. Central 95% interval coverage, mean interval width, and SR-minus-target calibration error are reported separately. LSTM service upper bounds use validation residual quantiles at the requested one-sided probability. They are a comparison calibration, not a probabilistic model output.

## Repeats And Runtime

The default is three seeds, `seed + run_index`, with TensorFlow and NumPy seeded per run. Raw rows are preserved by run ID; model, quantile, service-level, runtime, transfer, and uncertainty summaries have aggregate CSVs reporting mean, median, population standard deviation, minimum, maximum, and run count where numeric observations exist.

Training time measures the model fit call. Fine-tuning time measures the target fit separately and is zero for non-fine-tuned runs. Single-sample inference latency is measured after a warm-up and reported as mean, median, and population standard deviation across the requested repetitions. Parameter count comes from Keras. Model size is the saved weight-file size. Peak memory is the process' maximum resident-set high-water mark, not an isolated per-model delta; it is available on Linux through `resource`.

## Transfer Learning

The transfer runner trains an HBNN source model, saves and reloads its weights, and evaluates on the target test portion. It includes:

- All: all 12 processed traces are sources, including the target's training history.
- All-but-one: the target is excluded from source training (zero-shot).
- Fine-tuned variants: the source weights are loaded and fit on target training/validation windows; target test rows remain unseen.
- Same-distribution: a Google 2019 target uses the other Google 2019 traces as source.
- Different-distribution: Google 2019 traces are sources and Google 2011/Alibaba targets are evaluated zero-shot or fine-tuned.

The output includes source/target groups, historical-data and fine-tuning flags, point/probabilistic/service-level results, and training/fine-tuning/inference times. This is a configurable representative scenario set, not every possible source-target subset or an exhaustive search.

## Statistics And Limitations

No Diebold-Mariano p-values are currently generated. The paper's test compares paired forecasts and depends on aligned per-time errors, a specified loss differential, forecast horizon, and a variance estimator. The pipeline preserves target timestamps and per-sample predictions needed for a carefully specified follow-up, but three quick seeds alone are not a reason to report significance. No statistical significance claim should be inferred from aggregate CSVs.

The quick run is a smoke/reproduction subset, not a paper result. Differences from the article can arise from CPU-only execution, fewer windows/epochs/seeds, Python 3.12 and newer TensorFlow/TFP APIs, altered batch/patience limits, and the corrected target-index split. The paper's GPU-scale sweep and full Talos search are not performed by default. A full CPU run can take substantial time and should be started sequentially, not in parallel.