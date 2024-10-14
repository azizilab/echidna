# echidna.tools.eval.py

import warnings
from anndata._core.views import ImplicitModificationWarning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d

import torch
import pyro
import pyro.distributions as dist

from echidna.tools.custom_dist import TruncatedNormal
from echidna.tools.housekeeping import load_model
from echidna.tools.utils import _custom_sort
from echidna.tools.data import create_z_pi, filter_low_var_genes
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
    
    samples = [sample_funcs[v](adata, **kwargs) for v in variable]
    if len(samples) == 1: return samples[0]
    return samples

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
    return torch.nn.functional.softplus(eta_samples)

def sample_cov(adata, num_samples=(1,)):
    """Sample a covariance matrix from the posterior distribution.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    num_samples : int, optional
        The number of samples to draw from the posterior distribution.
        By default, draw a single sample.

    Returns
    -------
    covariance : torch.Tensor
        A covariance matrix sampled from the posterior distribution.
        The covariance matrix is of shape `(num_genes, num_genes)`.
    """
    model = load_model(adata)
    config = model.config
    
    # Sample a Cholesky decomposition of the covariance matrix
    correlation_loc = pyro.param("corr_loc")
    correlation_scale = pyro.param("corr_scale")
    correlation_dist = dist.MultivariateNormal(correlation_loc, torch.diag(correlation_scale))
    transformed_dist = dist.TransformedDistribution(correlation_dist, dist.transforms.CorrCholeskyTransform())
    chol_samples = transformed_dist.sample(num_samples).squeeze()
    
    # Sample the scale for the covariance matrix
    scale_shape = pyro.param("scale_shape")
    scale_rate = pyro.param("scale_rate")
    scale = scale_shape / scale_rate
    
    # Compute the covariance matrix
    if not config.inverse_gamma:
        scale = scale[:, None]
    else:
        scale = 1 / scale[:, None]
    covariance = chol_samples * torch.sqrt(scale)
    covariance = covariance @ covariance.T
    
    return covariance

def normalize_cov(cov_matrix):
    """Normalize a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov_matrix : torch.Tensor, shape (n_features, n_features)
        Covariance matrix.

    Returns
    -------
    corr_matrix : torch.Tensor, shape (n_features, n_features)
        Correlation matrix.
    """
    std_dev = torch.sqrt(torch.diag(cov_matrix))
    outer_std_dev = std_dev[:, None] * std_dev[None, :]
    corr_matrix = cov_matrix / outer_std_dev
    corr_matrix.diagonal(0).fill_(1)
    return corr_matrix

def mahalanobis_distance_matrix(eta):
    cov_matrix = torch.cov(eta)
    cov_inv = torch.linalg.inv(cov_matrix)
    
    num_vectors = eta.size(1)
    distance_matrix = torch.zeros((num_vectors, num_vectors), device=eta.device)
    
    for i in range(num_vectors):
        diff_i = eta.T - eta.T[i]  # Broadcasting difference for row i
        distance_matrix[i] = torch.sqrt((diff_i @ cov_inv @ diff_i.T).diag())
    
    return distance_matrix

def distance_matrix_helper(eta, similarity_metric):
    if similarity_metric == "smoothed_corr":
        eta = eta.cpu().detach().numpy()
        eta = gaussian_filter1d(eta, sigma=6, axis=1, radius=8)
        return pdist(eta, metric="correlation")
    elif similarity_metric == "cov":
        return torch.cov(eta).cpu().detach().numpy()
    elif similarity_metric == "corr":
        eta_corr = torch.corrcoef(eta).cpu().detach().numpy()
        return 1 - eta_corr
    elif similarity_metric == "mahalanobis":
        return mahalanobis_distance_matrix(eta).cpu().detach().numpy()
    else:
        raise ValueError(
            "Invalid similarity_metric. Use `smoothed_corr`, `cov`, `corr`."
        )

def eta_tree_elbow_thresholding(
    eta,
    similarity_metric,
    plot_dendrogram: bool=False,
    plot_elbow: bool=False
):
    dist_mat = distance_matrix_helper(eta, similarity_metric)
    link_metric = "ward" if similarity_metric == "smoothed_corr" else "average"
    Z = linkage(dist_mat, link_metric)
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
    similarity_metric,
    frac=0.7, 
    dist_matrix=False,
    plot_dendrogram: bool=False,
):
    dist_mat = distance_matrix_helper(eta, similarity_metric)
    link_metric = "ward" if similarity_metric == "smoothed_corr" else "average"
    if dist_matrix:
        Z = linkage(squareform(dist_mat), link_metric)
    else:
        Z = linkage(dist_mat, link_metric)
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
    similarity_metric,
    thres: float, 
    plot_dendrogram: bool=False
):
    dist_mat = distance_matrix_helper(eta, similarity_metric)
    link_metric = "ward" if similarity_metric == "smoothed_corr" else "average"
    Z = linkage(dist_mat, link_metric)
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
    color_dict = dict(zip(
        dn.get("leaves"), dn.get("leaves_color_list")
    ))

    if "echidna" not in adata.uns:
        raise ValueError(
            "No echidna model has been saved for this AnnData object."
        )
    cluster_label = adata.uns["echidna"]["config"]["clusters"]
    hier_colors = [color_dict[int(i)] for i in adata.obs[cluster_label]]

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
    metric: str=None,
    threshold: float=0.,
):
    echidna = load_model(adata)
    metric = "smoothed_corr" if metric is None else metric
    adata.uns["echidna"]["save_data"]["dendrogram_metric"] = metric
    
    # If a threshold is set, use that threshold
    if threshold > 0.:
        method = "manual"
    
    adata.uns["echidna"]["save_data"]["dendrogram_method"] = method
    
    if method == "elbow":
        dn = eta_tree_elbow_thresholding(
            echidna.eta_ground_truth, similarity_metric=metric
        )
    elif method == "cophenetic":
        dn = eta_tree_cophenetic_thresholding(
            echidna.eta_ground_truth, similarity_metric=metric
        )
    else:
        adata.uns["echidna"]["save_data"]["threshold"] = threshold
        if threshold == 0.:
            logger.warning(
                "If not using `elbow` or `cophenetic` method, "
                "you must set `threshold` manually. Default `threshold=0`."
            )
        dn = eta_tree(
            echidna.eta_ground_truth, similarity_metric=metric, thres=threshold
        )
    
    assign_clones(dn, adata)
    logger.info(
        "Added `.obs['echidna_clones']`: the learned clones from eta."
    )
    del echidna