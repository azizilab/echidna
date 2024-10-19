# Echidna
Mapping genotype to phenotype through joint probabilistic modeling of single-cell gene expression and chromosomal copy number variation.


![Echidna](https://github.com/azizilab/echidna3/assets/73508804/cd5588f6-ab6f-4411-8f6a-47a9e4e9e6d5)


# Echidna Configuration Settings

## .OBS LABELS

| Setting        | Type   | Default         | Description                               |
|----------------|--------|-----------------|-------------------------------------------|
| `timepoint_label` | `str`  | `"timepoint"`    | Label for timepoints in the data.         |
| `counts_layer`    | `str`  | `"counts"`       | Name of the counts layer in the data.     |
| `clusters`        | `str`  | `"leiden"`       | Clustering method used in the data.       |

## TRAINING PARAMETERS

| Setting         | Type   | Default         | Description                                               |
|-----------------|--------|-----------------|-----------------------------------------------------------|
| `seed`          | `int`  | `42`            | Random seed for reproducibility.                          |
| `n_steps`       | `int`  | `10000`         | Maximum number of steps for Stochastic Variational Inference (SVI). |
| `learning_rate` | `float`| `0.1`           | Learning rate for the Adam optimizer.                     |
| `val_split`     | `float`| `0.1`           | Percentage of training data to use for validation.         |
| `patience`      | `int`  | `30`            | Early stopping patience (set to >0 to enable early stopping). |
| `device`        | `str`  | `"cuda" if is_available() else "cpu"` | Device to use for training (GPU if available, otherwise CPU). |
| `verbose`       | `bool` | `True`          | Whether to enable logging output.                         |

## MODEL HYPERPARAMETERS

| Setting              | Type    | Default   | Description                                                                        |
|----------------------|---------|-----------|------------------------------------------------------------------------------------|
| `inverse_gamma`       | `bool`  | `False`   | Whether to use inverse gamma for noisier data.                                     |
| `lkj_concentration`   | `float` | `1.0`     | Concentration parameter of LKJ prior. Values < 1.0 result in more diagonal covariance matrices. |
| `q_shape_rate_scaler` | `float` | `10.0`    | Scaler for the shape and rate parameters of the covariance diagonal for variational inference. |
| `q_corr_init`         | `float` | `0.01`    | Initial scale of the variational correlation.                                      |
| `q_cov_scaler`        | `float` | `0.01`    | Scaler for the covariance of the variational correlation.                          |
| `eta_mean_init`       | `float` | `2.0`     | Initial mean value for the eta parameter.                                          |
| `eps`                 | `float` | `5e-3`    | Small constant added to the diagonal to ensure positive definiteness (PD).         |
