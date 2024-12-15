# echidna.tools.housekeeping.py

from datetime import datetime
import logging
import os, gc

import numpy as np

import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from echidna.tools.utils import EchidnaConfig
from echidna.tools.model import Echidna
from echidna.utils import (
    ECHIDNA_GLOBALS,
    create_echidna_uns_key,
    get_logger,
)

logger = get_logger(__name__)

def set_posteriors(echidna, data):
    """
    Set model posteriors after training
    """
    echidna.eta_posterior = eta_posterior_estimates(echidna, data)
    echidna.c_posterior = c_posterior_estimates(eta=echidna.eta_posterior, mt=echidna.config._is_multi)
    echidna.cov_posterior = cov_posterior_estimate(inverse_gamma=echidna.config.inverse_gamma)
    echidna.corr_posterior = normalize_cov(echidna.cov_posterior)
    echidna.library_size = data[0].sum(-1, keepdim=True) * 1e-5
    return echidna

def get_learned_params(echidna, data):
    """
    Function to retrive the learned parameters for one pass
    """
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(*data)
    trained_model = pyro.poutine.replay(echidna.model, trace=guide_trace)
    trained_trace = pyro.poutine.trace(trained_model).get_trace(*data)
    params = trained_trace.nodes
    return params

def eta_posterior_estimates(echidna, data, num_samples=(int(1e3),)):
    """
    Posterior mean of eta
    """
    if isinstance(num_samples, tuple):
        num_samples = num_samples[0]
    X, _, pi, _ = data
    eta = torch.zeros([pi.shape[-1], X.shape[-1]])
    for _ in range(num_samples):
        params = get_learned_params(echidna, data)
        eta += F.softplus(params['eta']['value'].T)
    eta /= num_samples
    eta = eta.to(echidna.config.device)
    return eta

def c_posterior_estimates(eta, mt=True):
    """
    Posterior mean of c. Takes in posterior mean of eta
    """
    c = None
    if mt:
        c_shape = pyro.param('c_shape').squeeze(1)
        c = c_shape.unsqueeze(1) * eta.unsqueeze(0)
    else:
        c_shape = pyro.param('c_shape')
        c = c_shape * eta
    return c

def cov_posterior_estimate(inverse_gamma=False, num_samples=(int(1e3),)):
    """
    Posterior mean of covariance
    """
    corr_loc = pyro.param("corr_loc")
    corr_scale = pyro.param("corr_scale")
    corr_cov = torch.diag(corr_scale)
    corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
    transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
    chol_samples = transformed_dist.sample(num_samples)
    L_shape = pyro.param('scale_shape')
    L_rate = pyro.param('scale_rate')
    L = L_shape/L_rate
    
    scale = L[:, None] if not inverse_gamma else 1/L[:, None]
    cov = chol_samples.mean(0) * torch.sqrt(scale)
    cov = cov@cov.T
    
    return cov

def normalize_cov(cov):
    """Posterior mean of correlation matrix. Takes in estimated covariance."""
    std_dev = torch.sqrt(torch.diag(cov))
    outer_std_dev = torch.outer(std_dev, std_dev)
    corr_matrix = cov / outer_std_dev
    corr_matrix.fill_diagonal_(1)  # Set diagonal elements to 1
    return corr_matrix

def save_model(adata, model, overwrite=False, simulation=False):
    """
    Modified from Decipher with author permission:
    Achille Nazaret, https://github.com/azizilab/decipher/blob/main/decipher/tools/_decipher/data.py
    """
    create_echidna_uns_key(adata)
    
    run_id_key = "run_id_sim" if simulation else "run_id"
    run_id_key_hist = "run_id_sim_history" if simulation else "run_id_history"
    
    if run_id_key_hist not in adata.uns["echidna"]:
        adata.uns["echidna"][run_id_key_hist] = []
    else:
        run_hist = adata.uns["echidna"][run_id_key_hist]
        if isinstance(run_hist, np.ndarray):
            adata.uns["echidna"][run_id_key_hist] = list(run_hist)
    sim = "" if not simulation else "simulation "
    if run_id_key not in adata.uns["echidna"] or not overwrite:
        adata.uns["echidna"][run_id_key] = datetime.now().strftime("%Y%m%d-%H%M%S")
        adata.uns["echidna"][run_id_key_hist].append(adata.uns["echidna"][run_id_key])
        logging.info(f"Saving echidna {sim}model with run_id {adata.uns['echidna'][run_id_key]}.")
    else:
        logging.info(f"Overwriting existing echidna {sim}model.")

    model_run_id = adata.uns["echidna"][run_id_key]
    save_folder = ECHIDNA_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    
    torch.save(model, os.path.join(full_path, "echidna_model.pt"))
    
    param_store = pyro.get_param_store()
    param_dict = {name: param_store[name].detach().cpu() for name in param_store.keys()}
    torch.save(param_dict, os.path.join(full_path, "echidna_model_param_store.pt"))
    
    adata.uns["echidna"]["config"] = model.config.to_dict()

def load_model(adata, save_folder=None, simulation=False):
    """
    Modified from Decipher with author permission:
    Achille Nazaret, https://github.com/azizilab/decipher/blob/main/decipher/tools/_decipher/data.py
    
    Load an echidna model whose name is stored in the given AnnData.

    `adata.uns["echidna"]["run_id"]` must be set to the name of the echidna model to load.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    
    save_folder: str
        The location of saved model parameters. Default to None.

    Returns
    -------
    model : Echidna
        The echidna model.
    """
    create_echidna_uns_key(adata)
    
    run_id_key = "run_id_sim" if simulation else "run_id"
    run_id_key_hist = "run_id_sim_history" if simulation else "run_id_history"
    
    if run_id_key not in adata.uns["echidna"]:
        sim = "" if not simulation else "simulation "
        raise ValueError(f"No echidna {sim}model has been saved for this AnnData object.")

    model_config = EchidnaConfig(**adata.uns["echidna"]["config"])
    model = Echidna(model_config)
    model_run_id = adata.uns["echidna"][run_id_key]
    save_folder_path = ECHIDNA_GLOBALS["save_folder"] if save_folder is None else save_folder
    full_path = os.path.join(save_folder_path, model_run_id) if save_folder is None else save_folder
    model = torch.load(os.path.join(full_path, "echidna_model.pt"), weights_only=False)
    
    pyro.clear_param_store()
    gc.collect()
    torch.cuda.empty_cache()
    param_store = pyro.get_param_store()
    param_dict = torch.load(os.path.join(full_path, "echidna_model_param_store.pt"), weights_only=False)
    for name, param in param_dict.items():
        if name in param_store:
            param_store[name] = param.to(model.config.device)
        else:
            pyro.param(name, param.to(model.config.device))

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(model.config.device)

    return model