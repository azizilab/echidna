import pyro
import scanpy as sc
import pandas as pd
import numpy as np
import torch


class ModelConfig:
    num_genes: int = 500
    num_cells: int = 500
    num_timepoints: int = 2
    num_clusters: int = 10
    # scaler for the shape and rate parameters of covariance diag for variational inference
    q_shape_rate_scaler: float = 10.0
    # initialize the scale of variational correlation
    q_corr_init: float = 0.01

# Load scRNA, isolate tumor, recluster and filter out small clusters
def read_X(path, tumor_only=True, leiden_res=0.5, thres=20):
    X = sc.read_h5ad(path)
    if tumor_only:
        X = X[X.obs['celltype_bped_main'] == 'Melanocytes']
        sc.tl.leiden(X, resolution=leiden_res)
        cluster_counts = X.obs['leiden'].value_counts()
        threshold = thres
        small_clusters = cluster_counts[cluster_counts <= threshold].index.tolist()
        X = X[~X.obs['leiden'].isin(small_clusters)].copy()
        sc.pp.neighbors(X)
        sc.tl.leiden(X, resolution=leiden_res)
    return X

# Load WGS, filter out nans
def read_W(path):
    Wdf = pd.read_csv(path)
    Wdf = Wdf.set_index('gene')
    Wdf = Wdf[Wdf >= 1].dropna()
    return Wdf

# Generate pi for one timepoint
def format_pi_one_timepoint(X, timepoint, num_clusters):
    clusters = np.unique(np.array(X[X.obs['condition'] == timepoint].obs['leiden'].values))
    z_obs_series = X[X.obs['condition'] == timepoint].obs['leiden'].values
    pi_obs_series = np.unique(z_obs_series, return_counts=True)[1] / len(z_obs_series)
    clst_dict = dict(zip(clusters, pi_obs_series))
    pi_obs_t = torch.zeros(num_clusters)
    for k, v in clst_dict.items():
        pi_obs_t[int(k)] = v
    return pi_obs_t


# prepare input for echidna
def prepare_input(X, W, sample_name, timepoints, n_subsamples, device):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    
    matched_genes = X.var.index.intersection(W.index)
    W = W.loc[matched_genes]
    X = X[:, matched_genes]
    num_clusters = len(np.unique(np.array(X.obs['leiden'].values)))

    # format single-timestep input
    if len(timepoints) == 1:
        X_obs = torch.from_numpy(X.layers['counts']).to(device)
        condition = sample_name + "_" + timepoints[0] + "_" + "count"
        W = W[condition]
        W_obs = torch.from_numpy(W.values).to(torch.float32).to(device)
        z_obs_series = X.obs['leiden'].values.astype(int)
        pi_obs_series = np.unique(z_obs_series, return_counts=True)[1] / len(z_obs_series)
        z_obs = torch.from_numpy(np.array(z_obs_series)).to(torch.int32).to(device)
        pi_obs = torch.from_numpy(pi_obs_series).to(torch.float32).to(device)
    
    # format multi-timestep input
    else:
        Xs, Zs, Pis = [], [], []
        conditions = []
        for timepoint in timepoints:
            condition = sample_name + "_" + timepoint + "_" + "count"
            conditions.append(condition)
            if X[X.obs['condition'] == timepoint].shape[0] == 0:
                raise ValueError("One of more timepoints do not exist in this sample")
            
            # get X at a timepoint
            X_t = sc.pp.subsample(X[X.obs['condition'] == timepoint], n_obs=n_subsamples, copy=True)
            X_t_obs = torch.from_numpy(X_t.layers['counts']).to(torch.float32).to(device)

            # get z at a timepoint
            z_t = X_t.obs['leiden'].values.astype(int)
            z_obs_t = torch.from_numpy(np.array(z_t)).to(torch.int32).to(device)

            # get pi at a timepoint
            pi_obs_t = format_pi_one_timepoint(X, timepoint, num_clusters)

            Zs.append(z_obs_t)
            Xs.append(X_t_obs)
            Pis.append(pi_obs_t)
        X_obs = torch.stack(Xs)
        z_obs = torch.stack(Zs)
        pi_obs = torch.stack(Pis)
        W_obs = torch.from_numpy(W[conditions].values.T).to(torch.float32).to(device)
    return X_obs, W_obs, z_obs, pi_obs




