# echidna.tools.utils.py

import logging, os, shutil
import dataclasses
from dataclasses import dataclass, field

import pandas as pd
import scanpy as sc

from torch.cuda import is_available

from echidna.utils import (
    get_logger,
    ECHIDNA_GLOBALS,
    create_echidna_uns_key,
)

logger = get_logger(__name__)

@dataclass(unsafe_hash=True)
class EchidnaConfig:
    ## DATA PARAMETERS
    num_genes: int = None
    num_cells: int = None
    num_timepoints: int = None
    num_clusters: int = None
    
    timepoint_label: str="timepoint"
    counts_layer: str="counts"
    _is_multi: bool=None

    ## TRAINING PARAMETERS
    seed: int=42
    # max steps of SVI
    n_steps: int=10000
    # learning rate for Adam optimizer
    learning_rate: float=.1
    # % of training set to use for validation
    val_split: float=.1
    # cluster label to use in adata.obs
    clusters: str="pheno_louvain"
    # early stopping if patience > 0
    patience: int=None
    # gpus if available
    device: str="cuda" if is_available() else "cpu"
    # logging
    verbose: bool=True
    
    ## MODEL HYPERPARAMETERS
    # Use inverse gamma for noiser data
    inverse_gamma: bool=False
    # concentration parameter of LKJ. <1.0 more diag
    lkj_concentration: float = 1.0
    # scaler for the shape and rate parameters of covariance diag for variational inference
    q_shape_rate_scaler: float = 10.0
    # initialize the scale of variational correlation
    q_corr_init: float = 0.01
    # scaler for the covariance of variational correlation
    q_cov_scaler: float = 0.01
    # initial mean of eta
    eta_mean_init: float = 2.0
    # constant add to diag to ensure PD
    eps: float = 5e-3
    
    def to_dict(self):
        res = dataclasses.asdict(self)
        return res
    
    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class EarlyStopping:
    """
    Borrowed from Decipher with author permission:
    Achille Nazaret, https://github.com/azizilab/decipher/blob/main/decipher/tools/utils.py
    
    Keeps track of when the loss does not improve after a given patience.
    Useful to stop training when the validation loss does not improve anymore.

    Parameters
    ----------
    patience : int
        How long to wait after the last validation loss improvement.

    Examples
    --------
    >>> n_epochs = 100
    >>> early_stopping = EarlyStopping(patience=5)
    >>> for epoch in range(n_epochs):
    >>>     # train
    >>>     validation_loss = ...
    >>>     if early_stopping(validation_loss):
    >>>         break
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.validation_loss_min = float("inf")

    def __call__(self, validation_loss):
        """Returns True if the training should stop."""
        if validation_loss < self.validation_loss_min:
            self.validation_loss_min = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def has_stopped(self):
        """Returns True if the stopping condition has been met."""
        return self.early_stop

def set_sort_order(adata, order):
    create_echidna_uns_key(adata)
    adata.uns["echidna"]["timepoint_order"] = {
        item: index for index, item in enumerate(order)
    }
    
def _custom_sort(items, order_dict):
    order_dict = {item: int(index) for item, index in order_dict.items()}
    default_order = len(order_dict)

    items_list = items.columns if isinstance(items, pd.DataFrame) else items
    sorted_list = sorted(items_list, key=lambda x: next((order_dict[sub] for sub in order_dict.keys() if sub in x), default_order))
    
    if isinstance(items, pd.DataFrame):
        return items.reindex(columns=sorted_list, fill_value=None)
    
    return sorted_list

def reset_echidna_memory():
    save_folder = ECHIDNA_GLOBALS["save_folder"]
    
    if os.path.exists(save_folder) and os.path.isdir(save_folder):
        shutil.rmtree(save_folder)
    logger.info("Cleared Echidna model saves.")
