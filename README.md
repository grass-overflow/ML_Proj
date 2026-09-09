# ML_Proj

# Forecasting Workload in Cloud Computing: Towards Uncertainty-Aware Predictions and Transfer Learning

Predicting future resource demand in Cloud Computing is essential for optimizing the trade-off between serving customers' requests efficiently and minimizing the provisioning cost. Modelling prediction uncertainty is also desirable to better inform the resource decision-making process, but research in this field is under-investigated. In this paper, we propose univariate and bivariate Bayesian deep learning models that provide predictions of future workload demand and its uncertainty. We run extensive experiments on Google and Alibaba clusters, where we first train our models with datasets from different cloud providers and compare them with LSTM-based baselines. Results show that modelling the uncertainty of predictions has a positive impact on performance, especially on service level metrics, because uncertainty quantification can be tailored to desired target service levels that are critical in cloud applications.=

## Python Dependencies
* arch                      5.1.0
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.12.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* tensorflow-probability    0.14.0

## Project Structure
* **hyperparams**: contains for each deep learning model the list of optimal hyperparameters found with Talos.
* **img**: contains output plot for predictions, models and loss function.
* **models**: contains the definition of statistical and deep learning models. One can train the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains for each statistical model the list of optimal parameters found.
* **res**: contains the results of the prediction
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the training phase.
* **time**: contains measurements of the time for training, fine-tuning and inference phases.
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

#### Train LSTM

```bash
python lstm_training.py
```

#### Train HBNN

```bash
python hbnn_training.py
```

#### Train LSTMD

```bash
python lstmd_training.py
```

##### Model types:
* univariate
* bivariate
* multiunivariate
* multibivariate

#### Methods:

```bash
training_talos
```

Run the hyperparameters search via Talos optimization library

```bash
training
```

Train a model from scratch, given a set of hyperparameters

```bash
load_and_predict
```

Load a pretrained model and make the prediction

```bash
load_and_tune
```

Load a pretrained model and tune the network on the specified dataset

Results:
<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/efe4a575-a4db-45b1-a48f-f86ebe68bd1d" />

<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/baae563e-7fd0-4126-b0b5-857ff01fafda" />

<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/4837e321-52c5-495d-80dd-30d829072b59" />

<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/5a002e8e-e943-409f-a9be-e113f3e4d98f" />

<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/24b50366-e656-41c1-9ba4-a3e7afd57d35" />

<img width="1156" height="470" alt="image" src="https://github.com/user-attachments/assets/3b9e00d0-abba-4d1c-847f-17cef497ed3f" />






