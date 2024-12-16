# Echidna
<img src=./echidna_logo.png width="100" />

 A Bayesian framework for quantifying gene dosage effect impacting phenotypic plasticity through integrating single-cell RNA sequencing (scRNA-seq) and bulk whole-genome sequencing (WGS) from a single or multiple time points. 


<img src=./echidna_concept.png width="800" />

# Install

Echidna is available on PyPI under the name `echidna-sc`.

## Step 1 (optional but recommended)
Create a conda environment with a recent Python version: conda create -n "echidna-env" python=3.10

## Step 2
Ensure you have the right version of torch installed for your device. See instructions on [pytorch.org](https://pytorch.org/get-started/locally/).

Echidna is available through pip via: `pip install echidna-sc`.

# Tutorial

There are four example notebooks:

1. 1-single-timepoint.ipynb
2. 2-multi-timepoint.ipynb
3. 3-infer-gene-dosage.ipynb
4. 4-echidna-model.ipynb

The notebooks are meant to be run sequentially, and they build off of each other.

Notebook 1 introduces you to the package - preparing your data, setting hyperparamters, performing posterior predictive checks - with data collected from a single point in time.

In notebook 2, we look at a multi-timepoint setting, where we have paired single-cell and WGS data collected over time. 

The saved model runs from notebook 2 will be used in notebook 3, where you will see how to infer amplifications and deletions by cluster of genes across a given genome. This notebook also shows you how to calculate and plot gene dosage effect with Echidna.

Notebook 4 is meant to show you how to do more custom work with the model. We package together many functions for your convenience, but this notebook will show you how to work directly with the model for the experiments not covered in the package. Some [Pyro](https://pyro.ai/) knowledge is assumed.

# Echidna Configuration Settings

## `.obs` Labels

| Setting        | Type   | Default         | Description                               |
|----------------|--------|-----------------|-------------------------------------------|
| `timepoint_label` | `str`  | `"timepoint"`    | Label for timepoints in the data.         |
| `counts_layer`    | `str`  | `"counts"`       | Name of the counts layer in the data.     |
| `clusters`        | `str`  | `"leiden"`       | Clustering method used in the data. This can also be celltype annotations, if you have them.       |

## Training Parameters

| Setting         | Type   | Default         | Description                                               |
|-----------------|--------|-----------------|-----------------------------------------------------------|
| `seed`          | `int`  | `42`            | Random seed for reproducibility.                          |
| `n_steps`       | `int`  | `10000`         | Maximum number of steps for Stochastic Variational Inference (SVI). |
| `learning_rate` | `float`| `0.1`           | Learning rate for the Adam optimizer.                     |
| `val_split`     | `float`| `0.1`           | Percentage of training data to use for validation.         |
| `patience`      | `int`  | `30`            | Early stopping patience (set to >0 to enable early stopping). |
| `device`        | `str`  | `"cuda" if is_available() else "cpu"` | Device to use for training (GPU if available, otherwise CPU). |
| `verbose`       | `bool` | `True`          | Whether to enable logging output.                         |

## Model Hyperparameters

| Setting              | Type    | Default   | Description                                                                        |
|----------------------|---------|-----------|------------------------------------------------------------------------------------|
| `inverse_gamma`       | `bool`  | `False`   | Whether to use inverse gamma for noisier data.                                     |
| `lkj_concentration`   | `float` | `1.0`     | Concentration parameter of LKJ prior. Values > 1.0 result in more diagonal covariance matrices. |
| `q_shape_rate_scaler` | `float` | `10.0`    | Scaler for the shape and rate parameters of the covariance diagonal for variational inference. |
| `q_corr_init`         | `float` | `0.01`    | Initial scale of the variational correlation.                                      |
| `q_cov_scaler`        | `float` | `0.01`    | Scaler for the covariance of the variational correlation.                          |
| `eta_mean_init`       | `float` | `2.0`     | Initial mean value for the eta parameter.                                          |
| `eps`                 | `float` | `5e-3`    | Small constant added to the diagonal to ensure positive definiteness (PD).         |
