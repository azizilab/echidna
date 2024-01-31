import pyro
import scanpy as sc
import pandas as pd
import numpy as np

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

# Function to retrive the learned parameters
def get_learned_params(echidna, X, W, pi, z):
    guide_trace = pyro.poutine.trace(echidna.guide).get_trace(X, W, pi, z)
    trained_model = pyro.poutine.replay(echidna.model, trace=guide_trace)
    trained_trace = pyro.poutine.trace(trained_model).get_trace(
            X, W, pi, z
        )
    params = trained_trace.nodes
    return params



