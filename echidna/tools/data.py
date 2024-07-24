# echidna.tools.data.py

import pyro
import torch
import numpy as np
import pandas as pd
import scanpy as sc

from echidna.tools.utils import EchidnaConfig, _custom_sort
from echidna.utils import get_logger

logger = get_logger(__name__)

def pre_process(
    adata: sc.AnnData, 
    num_genes: int=None, 
    target_sum: float=None, 
    exclude_highly_expressed: bool=False, 
    n_comps: int=15, 
    phenograph_k: int=60, 
    n_neighbors: int=15,
    filepath: str=None,
    ) -> sc.AnnData:
    """Basic pre-processing pipeline. Choose parameters according to your data.
    
    Parameters
    ----------
    adata : sc.AnnData
        Unprocessed annotated data matrix.
    num_genes : int
        Number of highly expressed genes to use. Pass None if using
        all (recommended).
    target_sum : 
        Normalize to this total, defaults median library size.
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes.
    n_comps : int
        Number of principal components to use for PCA.
    phenograph_k : int
        Number of nearest neighbors to use in first step of graph
        construction.
    n_neighbors : int
        Number of nearest neighbors for UMAP.
    filepath : str
        If defined, will save the processed AnnData to the specified location.
        
    Returns
    -------
    adata : sc.AnnData
        Processed annotated data matrix.
    """
    from scipy.sparse import csr_matrix

    adata.X = adata.X.astype("float32")

    # highly variable genes
    if num_genes is not None:
        x_log = sc.pp.log1p(adata, copy=True, base=10)
        sc.pp.highly_variable_genes(x_log, n_top_genes=num_genes)
        adata = adata[:, x_log.var["highly_variable"]]

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.calculate_qc_metrics(adata, inplace=True, layer="counts")

    # store the current "total_counts" under original_total_counts, 
    # which will not automatically be updated by scanpy in subsequent filtering steps
    adata.obs["original_total_counts"] = adata.obs["total_counts"].copy()

    # log10 original library size
    adata.obs["log10_original_total_counts"] = np.log10(adata.obs["original_total_counts"]).copy()

    # Normalize by median library size
    if target_sum is None:
        target_sum = np.median(adata.obs["original_total_counts"])
    sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

    # log transform + 1 and updates adata.X
    sc.pp.log1p(adata)

    logger.info("Performing PCA...")
    sc.tl.pca(adata, n_comps=n_comps)
    logger.info("Calculating phenograph clusters...")
    sc.external.tl.phenograph(adata, clustering_algo="leiden", k=phenograph_k, seed=1)

    logger.info("Performing nearest neighbors search and calculating UMAP...")
    sc.pp.neighbors(adata, random_state=1, n_neighbors=n_neighbors)
    sc.tl.umap(adata, random_state=1)

    logger.info("Done processing.")

    for sparse_mtx in adata.obsp:
        adata.obsp[sparse_mtx] = csr_matrix(adata.obsp[sparse_mtx])
    if filepath is not None:
        adata.write_h5ad(filepath)
    return adata

def filter_low_var_genes(
    adata: sc.AnnData,
    var_threshold: float=0.01
) -> sc.AnnData:
    gene_filter = adata.X.var(axis=0) > var_threshold
    return adata[:, gene_filter]

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
    
    logger.info(
        "Added `.obs['echidna_split']`: the Echidna train/validation split.\n"
        f" {n_val} cells in validation set."
    )

    return adata

def create_z_pi(adata, config):
    config = config.to_dict() if isinstance(config, EchidnaConfig) else config
    if config["clusters"] not in adata.obs.columns:
        raise ValueError(f"{config['clusters']} clustering not in AnnData obs")
    if not bool(config["_is_multi"]):
        z_obs_series = adata.obs[config["clusters"]].values
        pi_obs_series = np.unique(z_obs_series, return_counts=True)[1] / len(z_obs_series)
        z_obs = torch.from_numpy(np.array(z_obs_series)).to(torch.float32).to(config["device"])
        pi_obs = torch.from_numpy(pi_obs_series).to(torch.float32).to(config["device"])
    else:
        adata_tmp = adata[adata.obs["echidna_split"]!="discard"].copy()
        timepoints = np.unique(adata_tmp.obs[config["timepoint_label"]])
        if "timepoint_order" in adata_tmp.uns["echidna"]:
            timepoints = _custom_sort(timepoints, adata.uns["echidna"]["timepoint_order"])
        z = []
        pi = []
        for t in timepoints:
            z_tmp = adata_tmp.obs[adata_tmp.obs[config["timepoint_label"]]==t][config["clusters"]].values
            pi_tmp = torch.zeros(int(config["num_clusters"]), dtype=torch.float32)
            indices, counts = np.unique(z_tmp, return_counts=True)
            pi_proportions = counts / len(z_tmp)
            for i, p in zip(indices, pi_proportions): pi_tmp[i] = p
            z.append(torch.tensor(z_tmp, dtype=torch.int64))
            pi.append(pi_tmp)
        z_obs = torch.stack(z).to(torch.float32).to(config["device"])
        pi_obs = torch.stack(pi).to(torch.float32).to(config["device"])
    return pi_obs, z_obs

def match_genes(adata, Wdf):
    """Matches genes between AnnData and W.

    Parameters
    ----------
        adata : sc.AnnData
            Annotated data matrix.
        Wdf : pd.DataFrame
            DataFrame containing copy number counts, indexed by genes.
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

def build_torch_tensors(adata, config):
    """
    Takes anndata and builds Torch tensors.
    """
    Wdf = adata.var[[c for c in adata.var.columns if "echidna_W_" in c]].dropna()
    if config._is_multi and "timepoint_order" in adata.uns["echidna"]:
        Wdf = _custom_sort(Wdf, adata.uns["echidna"]["timepoint_order"])
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
        tps = adata.obs[config.timepoint_label].unique()
        if "timepoint_order" in adata.uns["echidna"]:
            tps = _custom_sort(tps, adata.uns["echidna"]["timepoint_order"])
        x_list = [adata[tp == adata.obs[config.timepoint_label]].layers[config.counts_layer] for tp in tps]
        X_obs = torch.from_numpy(np.array(x_list)).to(torch.float32).to(config.device)
    elif not config._is_multi:
        W_obs = W_obs.flatten()
        X_obs = torch.from_numpy(adata.layers[config.counts_layer]).to(torch.float32).to(config.device)
    if config.clusters:
        pi_obs, z_obs = create_z_pi(adata, config)
        return X_obs, W_obs, pi_obs, z_obs
    return X_obs, W_obs