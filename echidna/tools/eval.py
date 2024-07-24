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
from echidna.tools.data import create_z_pi
from echidna.utils import get_logger

logger = get_logger(__name__)

def sample(adata, variable, **kwargs):
    if isinstance(variable, str):
        variable = [variable]
    elif not isinstance(variable, list):
        raise ValueError("`variable` must be either a string or a list.")
    
    allowed_vars = {"X", "W", "c", "cov", "eta"}
    
    invalid_vars = set(variable) - allowed_vars
    if invalid_vars:
        raise ValueError(
            f"The following `variable` values are not allowed: {invalid_vars}. "
            "Allowed values are (\"X\", \"W\", \"c\", \"eta\", \"cov\")."
        )
    
    sample_funcs = {
        "X": sample_X,
        "W": sample_W,
        "c": sample_c,
        "eta": sample_eta,
        "cov": sample_cov, 
    }
    
    return [sample_funcs[v](adata, **kwargs) for v in variable]

def sample_X(adata, num_cells=None, return_z=False):
    """Sample X given posterior estimates of c and eta
    """
    config = adata.uns["echidna"]["config"]

    echidna = load_model(adata)
    
    # num_cells cleaning and checks
    num_cells = int(config["num_cells"]) if num_cells is None else num_cells
    num_cells = num_cells[0] if isinstance(num_cells, (tuple, list, torch.Tensor, np.ndarray)) else num_cells
    if not isinstance(num_cells, int): raise ValueError("`num_cells` must be of type int.")
    
    # Check num_cells is within bounds
    ## Get the number of cells in each timepoint, which should
    ## be even across TP via the discard filter
    discard_filter = adata.obs["echidna_split"] != "discard"
    timepoint_label = config["timepoint_label"]
    arbitrary_tp_filter = adata.obs.loc[:, timepoint_label] == adata.obs.loc[:, timepoint_label].iloc[0]
    max_cells = adata.obs[discard_filter & arbitrary_tp_filter].shape[0]
    if num_cells > max_cells:
        raise ValueError(f"`num_cells` must be less than {max_cells}.")
        
    rng = np.random.default_rng(int(config["seed"]))
    
    z = []
    library_size = []
    timepoints = np.unique(adata.obs[config["timepoint_label"]])
    if "timepoint_order" in adata.uns["echidna"]: 
        timepoints = _custom_sort(timepoints, adata.uns["echidna"]["timepoint_order"])
    if "total_counts" not in adata.obs:
        adata.obs["total_counts"] = adata.layers[config["counts_layer"]].sum(-1)
        logger.info(
            "Added `.obs['total_counts']` : Library size, sum of counts"
            " across genes for each cell."
        )
    for i, t in enumerate(timepoints):
        timepoint_filter = adata.obs[config["timepoint_label"]]==t
        adata_tmp = adata[timepoint_filter, adata.var["echidna_matched_genes"]]
        
        cell_index = rng.choice(range(adata_tmp.shape[0]), num_cells, replace=False)
        z.append(
            torch.tensor(adata_tmp.obs.loc[
                adata_tmp.obs.index[cell_index], config["clusters"]
            ].values, dtype=torch.int64)
        )
        library_size.append(
            torch.tensor(
                adata_tmp.obs.loc[:, "total_counts"].iloc[cell_index].values,
                dtype=torch.float32,
            )
        )
    library_size = torch.stack(library_size) * 1e-5
    z = torch.stack(z)
    c = echidna.c_ground_truth
    if not bool(config["_is_multi"]): c = c.unsqueeze(0)
    eta = echidna.eta_ground_truth

    mean_scale = torch.mean(eta, axis=-1).repeat(int(config["num_genes"]),1).T
    c_scaled = c * mean_scale.unsqueeze(0)

    X_sample = []
    for t in range(int(config["num_timepoints"])):
        rate = c_scaled[t, z[t], :] * library_size.T[:,t].view(-1,1)
        X = dist.Poisson(rate).sample()
        X_sample.append(X)
    X_sample = torch.stack(X_sample, dim=0).squeeze()
    del echidna
    if return_z:
        return X_sample, z
    return X_sample

def sample_W(adata, num_samples=(1,)):
    """Sample W given posterior estimates of eta
    """
    config = adata.uns["echidna"]["config"]
    echidna = load_model(adata)
    
    pi, _ = create_z_pi(adata, config)
    
    W = TruncatedNormal(pi @ echidna.eta_ground_truth, 0.05, lower=0).sample(num_samples)
    W = W.squeeze()
    del echidna
    return W

def sample_c(adata, num_samples=(1,)) -> torch.Tensor:
    """Sample C from posterior and selec a target cluster for
    a given time point.
    """
    config = adata.uns["echidna"]["config"]
    echidna = load_model(adata)
    
    eta = dist.MultivariateNormal(
        echidna.eta_ground_truth.expand(config["num_clusters"], -1).T,
        covariance_matrix=echidna.cov_ground_truth
    ).sample()
    c_sample = dist.Gamma(pyro.param("c_shape"), 1/echidna.eta_ground_truth)
    c_sample = c_sample.sample(num_samples).squeeze()
    del echidna
    return c_sample

def sample_eta(adata, num_samples=(1,)):
    """Sample eta from posterior
    """
    config = adata.uns["echidna"]["config"]
    echidna = load_model(adata)
    
    eta_posterior = dist.MultivariateNormal(
        echidna.eta_ground_truth.expand(int(config["num_clusters"]), -1).T,
        covariance_matrix=echidna.cov_ground_truth
    )
    eta_samples = eta_posterior.sample(num_samples).squeeze()
    del echidna
    return eta_samples

def sample_cov(adata, num_samples=(1,)):
    """Return learned covariance across clusters.
    """
    echidna = load_model(adata)
    
    corr_loc = pyro.param("corr_loc")
    corr_scale = pyro.param("corr_scale")
    corr_cov = torch.diag(corr_scale)
    corr_dist = dist.MultivariateNormal(corr_loc, corr_cov)
    transformed_dist = dist.TransformedDistribution(corr_dist, dist.transforms.CorrCholeskyTransform())
    chol_samples = transformed_dist.sample(num_samples).squeeze()
    L_shape = pyro.param("scale_shape")
    L_rate = pyro.param("scale_rate")
    L = L_shape/L_rate
    
    scale = L[:, None] if not echidna.config.inverse_gamma else 1/L[:, None]
    cov = chol_samples * torch.sqrt(scale)
    cov = cov@cov.T
    
    del echidna
    
    return cov

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

def distance_matrix_helper(eta, method):
    if method == "cov":
        return torch.cov(eta).cpu().detach().numpy()
    elif method == "corr":
        eta_corr = torch.corrcoef(eta).cpu().detach().numpy()
        return 1 - eta_corr
    else:
        raise ValueError("Invalid method. Use 'cov' or 'corr'.")

def eta_tree_elbow_thresholding(
    eta,
    method,
    plot_dendrogram: bool=False,
    plot_elbow: bool=False
):
    dist_mat = distance_matrix_helper(eta, method)
    Z = linkage(dist_mat, "average")
    distance = Z[:, 2]
    differences = np.diff(distance)
    knee_point = np.argmax(differences)
    threshold = distance[knee_point]

    if plot_elbow:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(differences) + 1), differences, marker="o")
        ax.axvline(x=knee_point + 1, linestyle="--", label="ELBOW threshold", color="red")
        ax.legend()
        ax.set_xlabel("Merge Step")
        ax.set_ylabel("Difference in Distance")
        ax.set_title("Elbow Method")
        return fig
    elif plot_dendrogram:
        fig, ax = plt.subplots()
        dendrogram(Z, color_threshold=threshold, no_plot=False, ax=ax)
        ax.set_title("Echidna Clusters Hierarchy")
        return fig
    else: 
        logger.info(f"Dendrogram knee point: {knee_point + 1}")
        logger.info(f"Dendrogram threshold: {threshold:.4f}")
        return dendrogram(Z, color_threshold=threshold, no_plot=True)

def eta_tree_cophenetic_thresholding(
    eta, 
    method,
    frac=0.7, 
    dist_matrix=False,
    plot_dendrogram: bool=False,
):
    dist_mat = distance_matrix_helper(eta, method)
    if dist_matrix:
        Z = linkage(squareform(dist_mat), "average")
    else:
        Z = linkage(dist_mat, "average")
    coph_distances = cophenet(Z)
    max_coph_distance = np.max(coph_distances)
    threshold = frac * max_coph_distance
    
    if not plot_dendrogram:
        logger.info(f"Dendrogram threshold: {threshold:.4f}")
        return dendrogram(Z, color_threshold=threshold, no_plot=True)
    else:
        dendrogram(Z, color_threshold=threshold, no_plot=False)

def eta_tree(
    eta, 
    method,
    thres: float, 
    plot_dendrogram: bool=False
):
    dist_mat = distance_matrix_helper(eta, method)
    Z = linkage(dist_mat, "average")
    if not plot_dendrogram:
        return dendrogram(Z, color_threshold=thres, no_plot=True)
    else:
        dendrogram(Z, color_threshold=thres, no_plot=False)

def assign_clones(dn, adata):
    """Assign clones based on the covariance tree for each cell.

    Parameters
    ----------
    dn : dict
        A dictionary containing the leaves color list and leaves information.
    adata : sc.AnnData
        Annotated data matrix.
    """
    # Extract leaves color list and leaves
    color_dict = dict(zip(
        dn.get("leaves"), dn.get("leaves_color_list")
    ))

    # Check for echidna model in AnnData object
    if "echidna" not in adata.uns:
        raise ValueError(
            "No echidna model has been saved for this AnnData object."
        )

    # Get cluster label from echidna model
    cluster_label = adata.uns["echidna"]["config"]["clusters"]

    # Map hierarchical colors to cells
    hier_colors = [color_dict[int(i)] for i in adata.obs[cluster_label]]
    
    # Assign hierarchical colors to echidna clones
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        adata.obs["echidna_clones"] = pd.Series(
            hier_colors,
            index=adata.obs.index,
            dtype="category",
        )

def echidna_clones(
    adata,
    method: str="elbow",
    cov: bool=False,
    threshold: float=0.,
):
    echidna = load_model(adata)
    adata.uns["echidna"]["save_data"]["dendrogram_cov"] = cov
    cov_method = "cov" if cov else "corr"
    
    # If a threshold is set, use that threshold
    if threshold > 0.:
        method = "manual"
    
    adata.uns["echidna"]["save_data"]["dendrogram_method"] = method
    
    if method == "elbow":
        dn = eta_tree_elbow_thresholding(echidna.eta_ground_truth, method=cov_method)
    elif method == "cophenetic":
        dn = eta_tree_cophenetic_thresholding(echidna.eta_ground_truth, method=cov_method)
    else:
        adata.uns["echidna"]["save_data"]["threshold"] = threshold
        if threshold == 0.:
            logger.warning(
                "If not using `elbow` or `cophenetic` method, "
                "you must set `threshold` manually. Default `threshold=0`."
            )
        dn = eta_tree(echidna.eta_ground_truth, method=cov_method, thres=threshold)
    
    assign_clones(dn, adata)
    logger.info(
        "Added `.obs['echidna_clones']`: the learned clones from eta."
    )
    del echidna