# eval.py

import numpy as np
import pandas as pd
from typing import Tuple
import logging

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform

import pyro
import pyro.distributions as dist

import torch

from echidna.tools.custom_dist import TruncatedNormal
from echidna.tools.model import Echidna
from echidna.tools.housekeeping import get_learned_params, load_model

import warnings
from anndata._core.views import ImplicitModificationWarning

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)

def predictive_log_likelihood(echidna: Echidna, data: Tuple):
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(*data)
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(echidna.model, trace=guide_trace)
    ).get_trace(*data)
    log_prob = (model_trace.log_prob_sum() - guide_trace.log_prob_sum()).item()
    return log_prob

def sample_X(X, c, eta, z, library_size):
    """
    Sample X given posterior estimates of c and eta
    """
    mean_scale = torch.mean(eta,axis=1).repeat(X.shape[-1],1).T
    c_scaled = c * mean_scale
    rate = c_scaled[z] * library_size
    X_learned = dist.Poisson(rate).sample()
    X_learned = X_learned.cpu().detach().numpy()
    return X_learned

def sample_W(pi, eta):
    """
    Sample W given posterior estimates of eta
    """
    W = TruncatedNormal(pi @ eta, 0.05, lower=0).sample()
    return W.detach().cpu().numpy()

def sample_c(
    c_shape,
    c_rate,
    num_clusters,
    num_timepoints,
    target_dim,
    target_timepoint,
    sample_size=1000
) -> torch.Tensor:
    """
    Sample C from posterior and selec a target cluster for a
    given time point
    """
    c_shape = torch.stack([c_shape] * num_clusters, dim=1).squeeze()
    c_rate = torch.stack([c_rate] * num_timepoints, dim=0)
    c_posterior = dist.Gamma(c_shape, c_rate)
    c_samples = c_posterior.sample([sample_size])
    return c_samples[:, target_timepoint, target_dim, :]

# Sample eta from posterior select a target cluster
def sample_eta(eta_mean, cov, target_dim, sample_size=1000):
    eta_posterior = dist.MultivariateNormal(eta_mean, covariance_matrix=cov)
    samples = eta_posterior.sample([sample_size])
    return samples[:, :, target_dim]

# return learned covariance across clusters
def learned_cov(L, scale):
    L = L * torch.sqrt(scale[:, None])
    sigma = L @ L.T
    return sigma

# Return clone tree based on learned covariance
def eta_cov_tree(eta, thres, plot_dendrogram: bool=False):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    fig = plt.figure(figsize=(6, 3))
    dn = dendrogram(Z, color_threshold=thres, no_plot=not plot_dendrogram)
    return dn

# Return clone tree based on learned covariance and compute elbow-optimized cutoff
def eta_cov_tree_elbow_thresholding(
    eta,
    plot_dendrogram: bool=False,
    plot_elbow: bool=False
):
    Z = linkage(torch.cov(eta).cpu().detach().numpy(), 'average')
    distance = Z[:,  2]
    differences = np.diff(distance)
    knee_point = np.argmax(differences)
    threshold = distance[knee_point]

    if plot_elbow:
        plt.figure()
        plt.plot(range(1, len(differences) + 1), differences, marker='o')
        plt.axvline(x=knee_point + 1, linestyle='--', label='ELBOW threshold', color='red')
        plt.legend()
        plt.xlabel('Merge Step')
        plt.ylabel('Difference in Distance')
        plt.title('Elbow Method')
        plt.show()
    elif plot_dendrogram:
        dendrogram(Z, color_threshold=threshold, no_plot=False)
        plt.title('Echidna Clusters Hierarchy')
    else: 
        logger.info(f"Dendrogram knee point: {knee_point + 1}")
        logger.info(f"Dendrogram threshold: {threshold:.4f}")
        return dendrogram(Z, color_threshold=threshold, no_plot=True)

# Assign clones based on the covariance tree for each cell
def assign_clones(dn, X):
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

def eta_cov_tree_cophenetic_thresholding(mat, method='average', frac=0.7, dist_matrix=False):
    if dist_matrix:
        Z = linkage(squareform(mat), method)
    else:
        Z = linkage(torch.cov(mat).cpu().detach().numpy(), method)
    # Compute the cophenetic distances
    coph_distances = cophenet(Z)

    max_coph_distance = np.max(coph_distances)
    threshold = frac * max_coph_distance
    
    # Plot the dendrogram with the threshold
    dn = dendrogram(Z, color_threshold=threshold)
    
    return dn

def echidna_clones(adata):
    echidna = load_model(adata)
    dn = eta_cov_tree_elbow_thresholding(echidna.eta_ground_truth)
    assign_clones(dn, adata)
    logger.info(
        "Added `.obs['echidna_clones']`: the learned clones from eta."
    )