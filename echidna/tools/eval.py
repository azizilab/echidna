# echidna.tools.eval.py

from typing import Tuple
import logging
import warnings
from anndata._core.views import ImplicitModificationWarning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform

import torch
import pyro
import pyro.distributions as dist

from echidna.tools.custom_dist import TruncatedNormal
from echidna.tools.model import Echidna
from echidna.tools.housekeeping import get_learned_params, load_model
from echidna.tools.utils import _custom_sort
from echidna.utils import get_logger

logger = get_logger(__name__)

def sample(adata, variable):
    if variable not in ("X", "W", "c", "eta"):
        raise ValueError("`variable` must be one of or a list of (\"X\", \"W\", \"c\", \"eta\")")
        
    sample_funcs = {
        "X" : sample_X,
        "W" : sample_W,
        "c" : sample_c,
        "eta" : sample_eta,
        # "cov" : 
    }
    
    return sample_funcs[variable](adata)

def sample_X(adata, num_cells=None):
    """Sample X given posterior estimates of c and eta
    """
    # echidna = load_model(adata)
    timepoints = _custom_sort(np.unique(adata.obs[echidna.config.timepoint_label]))
    # cells_filter = adata.obs["echidna_split"] != "discard"
    num_cells = echidna.config.num_cells if num_cells is None else num_cells
    num_cells = num_cells[0] if isinstance(num_cells, (tuple, list, torch.Tensor, np.ndarray)) else num_cells
    if not isinstance(num_cells, int): raise ValueError("`num_cells` must be of type int.")
    
    rng = np.random.default_rng(echidna.config.seed)
    
    z = []
    library_size = []
    for i, t in enumerate(timepoints):
        timepoint_filter = adata.obs[echidna.config.timepoint_label]==t
        adata_tmp = adata[timepoint_filter, adata.var["echidna_matched_genes"]]
        
        cell_index = rng.choice(range(adata_tmp.shape[0]), num_cells, replace=False)
        z.append(
            torch.tensor(adata_tmp.obs.loc[
                adata_tmp.obs.index[cell_index], echidna.config.clusters
            ].values, dtype=torch.int64)
        )

        library_size.append(torch.tensor(adata_tmp.obs["total_counts"][cell_index].values, dtype=torch.float32))
    library_size = torch.stack(library_size)
    z = torch.stack(z)
    print(library_size.shape)
    c = echidna.c_ground_truth
    eta = echidna.eta_ground_truth

    mean_scale = torch.mean(eta, axis=1).repeat(echidna.config.num_genes,1).T
    c_scaled = c * mean_scale

    rate = c_scaled[z] * library_size
    X = dist.Poisson(rate).sample()
    X_sample.append(X.squeeze())
    
    X_sample = torch.stack(X_sample, dim=0)
    return X_sample.squeeze()

def sample_W(adata, num_samples=(1,)):
    """Sample W given posterior estimates of eta
    """
    echidna = load_model(adata)
    
    z = adata.obs[echidna.config.clusters].values
    pi = np.unique(z, return_counts=True)[1] / len(z)
    pi = torch.from_numpy(pi).to(torch.float32).to(echidna.config.device)
    
    W = TruncatedNormal(pi @ echidna.eta_ground_truth, 0.05, lower=0).sample(num_samples)
    W = W.squeeze()
    return W

def sample_c(adata, num_samples=(1,)) -> torch.Tensor:
    """Sample C from posterior and selec a target cluster for
    a given time point.
    """
    echidna = load_model(adata)
    
    eta = dist.MultivariateNormal(
        echidna.eta_ground_truth.expand(echidna.config.num_clusters, -1).T,
        covariance_matrix=echidna.cov_ground_truth
    ).sample()
    c_sample = dist.Gamma(pyro.param("c_shape"), 1/echidna.eta_ground_truth)
    c_sample = c_sample.sample(num_samples).squeeze()
    return c_sample

def sample_eta(adata, num_samples=(1,)):
    """Sample eta from posterior
    """
    echidna = load_model(adata)
    
    eta_posterior = dist.MultivariateNormal(
        echidna.eta_ground_truth.expand(echidna.config.num_clusters, -1).T,
        covariance_matrix=echidna.cov_ground_truth
    )
    eta_samples = eta_posterior.sample(num_samples).squeeze()
    return eta_samples

def learned_cov(L, scale):
    """Return learned covariance across clusters.
    """
    L = L * torch.sqrt(scale[:, None])
    sigma = L @ L.T
    return sigma

def mahalanobis_distance_matrix(data, cov_matrix):
    cov_inv = torch.inverse(cov_matrix)
    data_np = data.cpu().detach().numpy()
    cov_inv_np = cov_inv.cpu().detach().numpy()
    
    num_samples = data_np.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(num_samples):
            delta = data_np[:, i] - data_np[:, j]
            distance_matrix[i, j] = np.sqrt(np.dot(np.dot(delta, cov_inv_np), delta.T))
    
    return distance_matrix

def eta_cov_tree_elbow_thresholding(
    eta,
    plot_dendrogram: bool=False,
    plot_elbow: bool=False
):
    """Return clone tree based on learned covariance and compute
    elbow-optimized cutoff.
    
    Parameters
    ----------
    eta : torch.Tensor
    plot_dendrogram : bool
    plot_elbow : bool
    """
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    distance = Z[:,  2]
    differences = np.diff(distance)
    knee_point = np.argmax(differences)
    threshold = distance[knee_point]

    if plot_elbow:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(differences) + 1), differences, marker='o')
        ax.axvline(x=knee_point + 1, linestyle='--', label='ELBOW threshold', color='red')
        ax.legend()
        ax.set_xlabel('Merge Step')
        ax.set_ylabel('Difference in Distance')
        ax.set_title('Elbow Method')
        return fig
    elif plot_dendrogram:
        fig, ax = plt.subplots()
        dendrogram(Z, color_threshold=threshold, no_plot=False, ax=ax)
        ax.set_title('Echidna Clusters Hierarchy')
        return fig
    else: 
        logger.info(f"Dendrogram knee point: {knee_point + 1}")
        logger.info(f"Dendrogram threshold: {threshold:.4f}")
        return dendrogram(Z, color_threshold=threshold, no_plot=True)
    
def eta_cov_tree_cophenetic_thresholding(
    mat, 
    method='average', 
    frac=0.7, 
    dist_matrix=False,
    plot_dendrogram: bool=False,
):
    if dist_matrix:
        Z = linkage(squareform(mat), method)
    else:
        Z = linkage(torch.cov(mat).cpu().detach().numpy(), method)
    # Compute the cophenetic distances
    coph_distances = cophenet(Z)

    max_coph_distance = np.max(coph_distances)
    threshold = frac * max_coph_distance
    
    if not plot_dendrogram:
        return dendrogram(Z, color_threshold=threshold, no_plot=True)
    else:
        dendrogram(Z, color_threshold=threshold, no_plot=False)

def eta_cov_tree(
    eta, 
    thres: float, 
    plot_dendrogram: bool=False
):
    """Return clone tree based on learned covariance.
    """
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    if not plot_dendrogram:
        return dendrogram(Z, color_threshold=thres, no_plot=True)
    else:
        dendrogram(Z, color_threshold=thres, no_plot=False)

def assign_clones(dn, X):
    """Assign clones based on the covariance tree for each cell.
    """
    clst = dn.get('leaves_color_list')
    keys = dn.get('leaves')
    color_dict = pd.DataFrame(clst)
    color_dict.columns=['color']
    color_dict.index=keys
    if "echidna" not in X.uns:
        raise ValueError("No echidna model has been saved for this AnnData object.")
    cluster_label = X.uns["echidna"]["config"]["clusters"]
    hier_colors = [color_dict.loc[int(i)][0] for i in X.obs[cluster_label]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        X.obs["echidna_clones"] = hier_colors
        X.obs["echidna_clones"] = X.obs["echidna_clones"].astype("category")

def echidna_clones(adata, method="elbow", threshold=0.):
    echidna = load_model(adata)
    if threshold > 0.: method="manual"
    if method=="elbow":
        adata.uns["echidna"]["save_data"]["dendrogram_method"] = "elbow"
        dn = eta_cov_tree_elbow_thresholding(echidna.eta_ground_truth)
    elif method=="cophenetic":
        adata.uns["echidna"]["save_data"]["dendrogram_method"] = "cophenetic"
        dn = eta_cov_tree_cophenetic_thresholding(echidna.eta_ground_truth)
    else:
        adata.uns["echidna"]["save_data"]["dendrogram_method"] = "manual"
        adata.uns["echidna"]["save_data"]["threshold"] = threshold
        if threshold==0.: logger.warning(
            "If not using `elbow` or `cophenetic` method"
            ", you must set `threshold` manually. Default `threshold=0`."
        )
        dn = eta_cov_tree(echidna.eta_ground_truth, thres=threshold)
    assign_clones(dn, adata)
    logger.info(
        "Added `.obs['echidna_clones']`: the learned clones from eta."
    )