# echidna.tools.train.py

import os, gc
import pickle as pkl
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pyro
import pyro.optim as optim
import torch
from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

from echidna.tools.utils import (
    EchidnaConfig, 
    EarlyStopping,
    set_sort_order,
)
from echidna.tools.model import Echidna
from echidna.tools.eval import sample
from echidna.tools.housekeeping import save_model, set_posteriors
from echidna.tools.data import (
    build_torch_tensors,
    match_genes,
    train_val_split,
    create_z_pi,
)
from echidna.plot.post import plot_loss
from echidna.utils import get_logger

logger = get_logger(__name__)

def predictive_log_likelihood(echidna: Echidna, data: Tuple):
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(*data)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(echidna.model, trace=guide_trace)
    ).get_trace(*data)
    log_prob = (model_trace.log_prob_sum() - guide_trace.log_prob_sum()).item()
    return log_prob

def echidna_train(adata, Wdf, config=EchidnaConfig()):
    """
    Input
    -----
    adata: sc.Anndata
    Wdf: pd.DataFrame
    config: EchidnaConfig, optional
    
    Output
    ------
    """
    pyro.util.set_rng_seed(config.seed)
    
    num_timepoints = len(adata.obs[config.timepoint_label].unique())
    config.num_timepoints = num_timepoints if config.num_timepoints is None else config.num_timepoints
    config._is_multi = True if config.num_timepoints > 1 else False
    
    config.num_clusters = len(adata.obs[config.clusters].unique())
    
    adata = train_val_split(adata, config)
    
    match_genes(adata, Wdf)
    
    cluster_dtype = adata.obs.loc[:, config.clusters].dtype
    if not pd.api.types.is_integer_dtype(cluster_dtype):
        adata.obs.loc[:, config.clusters + "_categorical"] = adata.obs.loc[:, config.clusters].copy()
        logger.warning(
            f"`{config.clusters}` changed to `{config.clusters}_categorical`."
        )
    adata.obs.loc[:, config.clusters] = pd.Categorical(adata.obs.loc[:,config.clusters]).codes
    
    adata_match = adata[:, adata.var.echidna_matched_genes].copy()
    
    train_data = build_torch_tensors(adata_match[adata_match.obs["echidna_split"]=="train"], config)
    
    if config.val_split > 0:
        val_data = build_torch_tensors(adata_match[adata_match.obs["echidna_split"]=="validation"], config)
    else:
        val_data = None

    config.num_cells = train_data[0].shape[-2]
    config.num_genes = train_data[0].shape[-1]
        
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.device)
    
    echidna = train_loop(config, train_data, val_data)
    save_model(adata, echidna, overwrite=True)
    
    del echidna
    gc.collect()
    torch.cuda.empty_cache()
    
def train_loop(config, train_data, val_data):
    pyro.clear_param_store()
    echidna = Echidna(config)
    
    # optimizer = optim.CosineAnnealingLR({
    #     "optimizer": torch.optim.Adam,
    #       "optim_args": {"lr": echidna.config.learning_rate}
    #       , "T_max": 250
    # })
    optimizer = optim.Adam({
        "lr": echidna.config.learning_rate
    })
    
    elbo = Trace_ELBO()
    svi = SVI(echidna.model, echidna.guide, optimizer, loss=elbo)
    
    iterator = tqdm(range(echidna.config.n_steps)) if echidna.config.verbose else range(echidna.config.n_steps)
    best_loss = float("inf")
    
    if (
        echidna.config.patience is not None
        and echidna.config.patience > 0
    ):
        early_stopping = EarlyStopping(patience=echidna.config.patience)
    else:
        early_stopping = EarlyStopping(patience=int(1e30))
        
    patience_counter = 0
    training_loss, validation_loss = [], []
    for _ in iterator:
        try:
            train_elbo = svi.step(*train_data)
            if val_data is not None:
                val_elbo = -predictive_log_likelihood(echidna, val_data)
        except Exception as e:
            logger.error(e)
            echidna = set_posteriors(echidna, train_data)
            return echidna
        if val_data is not None:
            validation_loss.append(val_elbo)
        training_loss.append(train_elbo)
        if echidna.config.verbose:
            avg_loss = np.mean(training_loss[-8:])
            if val_data is not None:
                avg_val_loss = np.mean(validation_loss[-8:])
            else:
                avg_val_loss = 0.
            iterator.set_description(
                f"training loss: {avg_loss:.4f} | "
                f"validation loss: {avg_val_loss:.4f}"
            )
        if early_stopping(avg_val_loss) and val_data is not None:
            break
    if early_stopping.has_stopped() and echidna.config.verbose and val_data is not None:
        logger.info("Early stopping has been triggered.")
    
    echidna = set_posteriors(echidna, train_data)
    
    if echidna.config.verbose:
        plot_loss(training_loss, label="training", log_loss=True)
        if val_data is not None:
            plot_loss(validation_loss, label="validation", log_loss=True)
    
    return echidna

def simulate(adata, overwrite=True):
    """Simulates a training run using sampled X and W.
    
    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix with previous training run stored
        in `.uns['echidna']`.
    overwrite : bool
        Whether or not to overwrite previous simulation runs. Default True
        to save memory.
    """
    adata_tmp = adata[adata.obs["echidna_split"] != "discard"].copy()
    config = EchidnaConfig.from_dict(adata_tmp.uns["echidna"]["config"])
    
    X_sample, z_obs = sample(adata_tmp, "X", return_z=True)
    pi_obs = torch.zeros((config.num_timepoints, config.num_clusters))
    for t in range(config.num_timepoints):
        z_tmp = z_obs[t] if len(z_obs.shape) == 2 else z_obs
        counts = torch.bincount(z_obs[t], minlength=config.num_clusters)  # Count occurrences of each cluster
        pi_obs[t] = counts.float() / z_obs.shape[-1]
    z_obs = z_obs.squeeze()
    pi_obs = pi_obs.squeeze()

    W_sample = sample(adata_tmp, "W")
    
    num_cells = z_obs.shape[-1]
    num_val = int(float(config.val_split) * num_cells)
    
    rng = np.random.default_rng(config.seed)
    val_index = rng.choice(range(num_cells), num_val, replace=False)
    train_index = np.setdiff1d(range(num_cells), val_index)
    val_index = torch.from_numpy(val_index).long().to(config.device)
    train_index = torch.from_numpy(train_index).long().to(config.device)
    
    if not config._is_multi:
        z_obs_train = z_obs[train_index]
        z_obs_val = z_obs[val_index]
        X_sample_train = X_sample[train_index, :]
        X_sample_val = X_sample[val_index, :]
    else:
        z_obs_train = z_obs[:, train_index]
        z_obs_val = z_obs[:, val_index]
        X_sample_train = X_sample[:, train_index, :]
        X_sample_val = X_sample[:, val_index, :]
    
    train_data = (X_sample_train, W_sample, pi_obs, z_obs_train)
    val_data = (X_sample_val, W_sample, pi_obs, z_obs_val)

    echidna_sim = train_loop(config, train_data, val_data)
    save_model(adata, echidna_sim, overwrite=overwrite, simulation=True)
    
    del echidna_sim
    torch.cuda.empty_cache()