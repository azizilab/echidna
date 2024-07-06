# echidna.tools.train.py

import os, gc, logging
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
    _custom_sort, 
)
from echidna.tools.model import Echidna
from echidna.tools.housekeeping import save_model, set_posteriors
from echidna.plot.post import plot_loss
from echidna.utils import get_logger

logger = get_logger(__name__)

def train_val_split(adata, config):
    rng = np.random.default_rng(config.seed)
    
    tmp_idx, i = "index", 0
    while tmp_idx in adata.obs.columns:
        tmp_idx = f"index{i}"
        i+=1
    adata.obs.reset_index(names=tmp_idx, inplace=True)
    
    if config._is_multi:
        adata_vc = adata.obs[config.timepoint_label].value_counts()

        n_val = int(config.val_split * adata_vc.min())
        smallest_tp = adata_vc.index[adata_vc.argmin()]

        adata.obs["echidna_split"] = "train"
        for tp in adata_vc.index:
            tp_filter = adata.obs[config.timepoint_label] == tp
            cur_tp_index = adata.obs[tp_filter].index

            val_idx = rng.choice(cur_tp_index, n_val, replace=False)

            n_discard = len(cur_tp_index) - adata_vc.min()
            if n_discard > 0:
                train_idx = np.setdiff1d(cur_tp_index, val_idx)
                discard_idx = rng.choice(train_idx, n_discard, replace=False)
                adata.obs.loc[discard_idx, "echidna_split"] = "discard"

            adata.obs.loc[val_idx, "echidna_split"] = "validation"

        adata.obs["echidna_split"] = adata.obs["echidna_split"].astype("category")
    else:
        n_val = int(config.val_split * adata.shape[0])
        val_idx = rng.choice(adata.obs.index, n_val, replace=False)
        adata.obs["echidna_split"] = "train"
        adata.obs.loc[adata.obs.index[val_idx], "echidna_split"] = "validation"
        adata.obs["echidna_split"] = adata.obs["echidna_split"].astype("category")
        
    adata.obs.set_index(tmp_idx, inplace=True)
    
    logging.info(
        "Added `.obs['echidna_split']`: the Echidna train/validation split.\n"
        f" {n_val} cells in validation set."
    )

    return adata

def match_genes(adata, Wdf):
    """
    Matches genes between AnnData and W.

    input
        adata: AnnData or Dict single cell
        Wdf: pd.DataFrame copy number
    output
        (AnnData/Dict, pd.DataFrame)
    """
    Wdf.dropna(inplace=True)
    matched_genes = adata.var.index.intersection(Wdf.index)
    adata.var["echidna_matched_genes"] = np.where(adata.var.index.isin(matched_genes), True, False)
    logger.info("Added `.var[echidna_matched_genes]` : Labled True for genes contained in W.")
    if len(Wdf.shape) > 1:
        Wdf.columns = [f"echidna_W_{c}" if "echidna_W_" not in c else c for c in Wdf.columns]
        col_name = Wdf.columns
    elif len(Wdf.shape) == 1:
        if Wdf.name is None:
            Wdf.name = "echidna_W_count"
        else:
            Wdf.name = "echidna_W_" + Wdf.name if "echidna_W_" not in Wdf.name else Wdf.name
        col_name = [Wdf.name]
    if len(np.intersect1d(adata.var.columns, col_name)) == 0:
        adata.var = adata.var.merge(Wdf, left_index=True, right_index=True, how="left")
        for c in col_name:
            logger.info(f"Added `.var[{c}]` : CN entries for genes contained in W.")
        
def convert_torch(adata, config):
    """
    Takes anndata or dictionary of single-cell and the copy number data and converts to torch tensors.
    """
    Wdf = adata.var[[c for c in adata.var.columns if "echidna_W_" in c]].dropna()
    if config._is_multi:
        Wdf = _custom_sort(Wdf)
    W_obs = torch.from_numpy(Wdf.values).to(torch.float32).to(config.device)
    
    if W_obs.shape[-1] != config.num_timepoints:
        raise ValueError(
            "Number of W columns found in AnnData does not match"
            " number of timepoints, drop excess if needed."
            " Check columns in `.var` :", list(Wdf.columns)
        )
    
    if config._is_multi:
        W_obs = W_obs.T
        adata = adata[adata.obs["echidna_split"]!="discard"]
        tps = _custom_sort(adata.obs[config.timepoint_label].unique())
        x_list = [adata[tp == adata.obs[config.timepoint_label]].layers["counts"] for tp in tps]
        X_obs = torch.from_numpy(np.array(x_list)).to(torch.float32).to(config.device)
    elif not config._is_multi:
        W_obs = W_obs.flatten()
        X_obs = torch.from_numpy(adata.layers["counts"]).to(torch.float32).to(config.device)

    if config.clusters:
        if config.clusters not in adata.obs.columns:
            raise ValueError(f"{cluster_label} clustering not in AnnData obs")
        
        if not config._is_multi:
            z_obs_series = adata.obs[config.clusters].values
            pi_obs_series = np.unique(z_obs_series, return_counts=True)[1] / len(z_obs_series)
            z_obs = torch.from_numpy(np.array(z_obs_series)).to(torch.float32).to(config.device)
            pi_obs = torch.from_numpy(pi_obs_series).to(torch.float32).to(config.device)
        else:
            timepoints = _custom_sort(np.unique(adata.obs[config.timepoint_label]))
            z = []
            pi = []
            for t in timepoints:
                z_tmp = adata.obs[adata.obs[config.timepoint_label]==t][config.clusters].values
                pi_tmp = torch.zeros(config.num_clusters, dtype=torch.float32)
                indices, counts = np.unique(z_tmp, return_counts=True)
                pi_proportions = counts / len(z_tmp)
                for i, p in zip(indices, pi_proportions): pi_tmp[i] = p
                z.append(torch.tensor(z_tmp, dtype=torch.int64))
                pi.append(pi_tmp)
            z_obs = torch.stack(z).to(torch.float32).to(config.device)
            pi_obs = torch.stack(pi).to(torch.float32).to(config.device)
        return X_obs, W_obs, pi_obs, z_obs
    return X_obs, W_obs

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
    pyro.clear_param_store()
    
    pyro.util.set_rng_seed(config.seed)
    
    num_timepoints = len(adata.obs[config.timepoint_label].unique())
    config.num_timepoints = num_timepoints if config.num_timepoints is None else config.num_timepoints
    if config.num_timepoints > 1: config._is_multi = True
    
    config.num_clusters = len(adata.obs[config.clusters].unique())
    
    adata = train_val_split(adata, config)
    match_genes(adata, Wdf)
    
    adata_match = adata[:, adata.var.echidna_matched_genes].copy()
    
    train_data = convert_torch(adata_match[adata_match.obs["echidna_split"]=="train"], config)
    val_data = convert_torch(adata_match[adata_match.obs["echidna_split"]=="validation"], config)

    config.num_cells = train_data[0].shape[-2]
    config.num_genes = train_data[0].shape[-1]

    echidna = Echidna(config)
        
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(config.device)
    
    optimizer = optim.CosineAnnealingLR({
        "optimizer": torch.optim.Adam,
          "optim_args": {"lr": config.learning_rate}
          , "T_max": 250
    })
    elbo = Trace_ELBO()
    svi = SVI(echidna.model, echidna.guide, optimizer, loss=elbo)
    
    iterator = tqdm(range(config.n_steps)) if config.verbose else range(config.n_steps)
    best_loss = float("inf")
    
    if (
        echidna.config.patience is not None
        and config.patience > 0
    ):
        early_stopping = EarlyStopping(patience=config.patience)
    else:
        early_stopping = EarlyStopping(patience=int(1e30))
        
    patience_counter = 0
    training_loss, validation_loss = [], []
    for _ in iterator:
        try:
            train_elbo = svi.step(*train_data)
            val_elbo = -predictive_log_likelihood(echidna, val_data)
        except Exception as e:
            logger.error(e)
            echidna = set_posteriors(echidna, train_data)
            return echidna
        validation_loss.append(val_elbo)
        training_loss.append(train_elbo)
        if config.verbose:
            avg_loss = np.mean(training_loss[-8:])
            avg_val_loss = np.mean(validation_loss[-8:])
            iterator.set_description(
                f"training loss: {avg_loss:.4f} | "
                f"validation loss: {avg_val_loss:.4f}"
            )
        if early_stopping(avg_val_loss):
            break
    if early_stopping.has_stopped() and config.verbose:
        logger.info("Early stopping has been triggered.")
    
    echidna = set_posteriors(echidna, train_data)
    save_model(adata, echidna, overwrite=True)
    
    # adata.uns["echidna"]["save_data"]["eta_ground_truth"] = echidna.eta_ground_truth
    # adata.uns["echidna"]["save_data"]["c_ground_truth"] = echidna.c_ground_truth
    # adata.uns["echidna"]["save_data"]["cov_ground_truth"] = echidna.cov_ground_truth
    # logger.info(
    #     "Added posteriors to `.uns[echidna][save_data]`"
    #     ": c, eta, and cov posterior means."
    # )

    plot_loss(training_loss, label="training", log_loss=True)
    plot_loss(validation_loss, label="validation", log_loss=True)
    
    gc.collect()
    torch.cuda.empty_cache()