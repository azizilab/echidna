# echidna.tools.utils.py

import logging
import dataclasses
from dataclasses import dataclass

import pandas as pd
import scanpy as sc

from torch.cuda import is_available

from echidna.utils import get_logger, ECHIDNA_GLOBALS

logger = get_logger(__name__)

@dataclass(unsafe_hash=True)
class EchidnaConfig:
    ## DATA PARAMETERS
    num_genes: int = None
    num_cells: int = None
    num_timepoints: int = None
    num_clusters: int = None
    
    timepoint_label: str="timepoint"
    _is_multi: bool=None
        
    ## TRAINING PARAMETERS
    seed: int=42
    # max steps of SVI
    n_steps: int=1000
    # learning rate for Adam optimizer
    learning_rate: float=.1
    # % of training set to use for validation
    val_split: float=.1
    # cluster label to use in adata.obs
    clusters: str="pheno_louvain"
    # early stopping if patience > 0
    patience: int=30
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

def pre_process(xad: sc.AnnData
                 , num_genes: int = None
                 , target_sum: float = None
                 , exclude_highly_expressed: bool = False
                 , n_comps: int = 15
                 , phenograph_k: int = 60
                 , n_neighbors: int = 15
                ) -> sc.AnnData:
        """
        xad: sc.AnnData
        num_genes -- number of highly expressed genes to use. Pass None if using all, default 2**13 (8192)
        target_sum -- normalize to this total, default median library size (ie None for scanpy)
        exclude_highly_expressed -- whether to exclude highly expressed genes (default: False)
        n_comps -- number of principal components to use for PCA (default: 15)
        phenograph_k --  Number of nearest neighbors to use in first step of graph construction (default: 60)
        n_neighbors -- num nearest neighbors for umap (default: 15)
        """
        
        xad.X = xad.X.astype("float32")
        
        # highly variable genes
        if num_genes is not None:
            x_log = sc.pp.log1p(xad, copy=True, base=10)
            sc.pp.highly_variable_genes(x_log, n_top_genes=num_genes)
            xad = xad[:, x_log.var["highly_variable"]]
        
        if "counts" not in xad.layers:
            xad.layers["counts"] = xad.X.copy()
        sc.pp.calculate_qc_metrics(xad, inplace=True, layer="counts")
        
        # store the current "total_counts" under original_total_counts, which will not automatically be updated
        # by scanpy in subsequent filtering steps
        xad.obs["original_total_counts"] = xad.obs["total_counts"].copy()

        # log10 original library size
        xad.obs["log10_original_total_counts"] = log10(xad.obs["original_total_counts"]).copy()

        # Normalize by median library size
        if target_sum is None:
            target_sum = median(xad.obs["original_total_counts"])
        sc.pp.normalize_total(xad, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

        #log transform + 1 and updates adata.X
        sc.pp.log1p(xad)
        
        logger.info("Performing PCA...")
        sc.tl.pca(xad, n_comps=n_comps)
        logger.info("Calculating phenograph clusters...")
        sc.external.tl.phenograph(xad, clustering_algo="louvain", k=phenograph_k, seed=1)
        
        logger.info("Performing nearest neighbors search and calculating UMAP...")
        sc.pp.neighbors(xad, random_state=1, n_neighbors=n_neighbors)
        sc.tl.umap(xad, random_state=1)
        
        logger.info("Done processing.")
        
        for sparse_mtx in xad.obsp:
            xad.obsp[sparse_mtx] = csr_matrix(xad.obsp[sparse_mtx])
        
        return xad

def _custom_sort(items):
    order = ECHIDNA_GLOBALS["timepoint_order"]
    order_dict = {item: index for index, item in enumerate(order)}
    default_order = len(order)

    items_list = items.columns if isinstance(items, pd.DataFrame) else items
    sorted_list = sorted(items_list, key=lambda x: next((order_dict[sub] for sub in order if sub in x), default_order))
    
    if isinstance(items, pd.DataFrame):
        return items.reindex(columns=sorted_list, fill_value=None)
    
    return sorted_list