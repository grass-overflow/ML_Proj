# Forecasting Workload in Cloud Computing


## Models

- **LSTM** is a deterministic Conv1D/LSTM point forecaster trained with mean squared error.
- **HBNN** retains the Bayesian DenseVariational layer and probabilistic Gaussian output. Repeated Bayesian forward passes estimate epistemic variance from the variation in predicted means; average output variance estimates aleatoric variance; their sum is total predictive variance.
- **LSTMD** retains the deterministic Conv1D/LSTM network with a probabilistic Gaussian output. 

Each model supports univariate CPU or memory targets and bivariate CPU/memory output. Single-dataset and combined-dataset training are supported by the experiment runner. 

## Datasets And Splits
The 12 files are `gc11`, `gc19_a` through `gc19_h`, `ali18`, `ali20_c`, and `ali20_g`. Each has `time`, `avgcpu`, and `avgmem` columns at five-minute intervals. 

The runner uses 288 observations as input and a horizon of two five-minute steps from the final observation. Windows are assigned by the timestamp of their target: the first 80% of target rows are train/validation and the remaining 20% are test. The last 20% of the pre-test windows are validation, leaving approximately 64% fit, 16% validation, and 20% test. Test inputs can include earlier history, but no test target is used for training, model selection, or interval calibration. Multi-dataset sources are concatenated in chronological trace order without shuffling.


## Install

Python 3.9–3.12 is supported by the pinned CPU dependency set. A normal isolated install is preferred:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

```bash
python3 -m pip install --target .python_packages -r requirements-cpu.txt
PYTHONPATH=.python_packages python3 run_experiments.py --mode quick --runs 1
```

## Run Experiments

Quick mode runs LSTM, HBNN, and LSTMD on `gc19_a` using a bounded sample, at most three epochs, and a batch size up to 64. It defaults to three seeds; pass `--runs 1` for the smallest smoke run.

```bash
python run_experiments.py --mode quick --runs 1
```

Other workflows:

```bash
python run_experiments.py --mode runtime --runs 1
python run_experiments.py --mode model_comparison --runs 1 --epochs 15
python run_experiments.py --mode transfer_learning --runs 1 --max-transfer-targets 1
python run_experiments.py --mode full --runs 3
```


## Metrics

- Point metrics: MSE, MAE.
- Service levels: SR is the fraction of actual values at or below the one-sided service-level upper quantile. TPR is `100 * sum(upper bound) / sum(actual demand)`. Central 95% interval coverage, interval width, and SR calibration error are reported separately.
- LSTM comparison intervals use empirical validation residual quantiles. They are not model uncertainty and are not used to label LSTM as uncertainty-aware.
- Runtime: training and fine-tuning wall time, repeated single-sample latency (mean/median/std), parameter count, serialized weight size, and process peak resident memory high-water mark.
rows for S-U, M-U, and M-B are references without corresponding run metrics.
