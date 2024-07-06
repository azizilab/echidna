# echidna.tools.post.py

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import logging

from echidna.tools.eval import (
    eta_cov_tree_elbow_thresholding,
    eta_cov_tree_cophenetic_thresholding,
    eta_cov_tree,
)
from echidna.tools.housekeeping import load_model
from echidna.plot.utils import save_figure, activate_plot_settings
from echidna.utils import get_logger

logger = get_logger(__name__)

def plot_loss(losses: list, label: str, log_loss: bool=False):
    activate_plot_settings()
    fig,ax = plt.subplots(1, 2, figsize=(12,4), sharey=False)
    if log_loss:
        sns.lineplot(np.log10(losses), ax=ax[0])
    else:
        sns.lineplot(losses, ax=ax[0])
    ax[0].set_title(f"{label} loss")
    ax[0].set_xlabel("steps")

    sns.lineplot(np.diff(losses), ax=ax[1])
    ax[1].set_title("step delta")
    ax[1].set_xlabel("steps")

    plt.show()

def dendrogram(adata, elbow: bool=False, filepath: str=None):
    activate_plot_settings()
    echidna = load_model(adata)
    try:
        method = adata.uns["echidna"]["save_data"]["dendrogram_method"]
    except Exception as e:
        logger.error(f"Must run `ec.tl.echidna_clones` first. {e}")
    if elbow==True and method!="elbow": logger.warning("`elbow=True` only applies to `method=\"elbow\"`.")
    if method=="elbow":
        fig = eta_cov_tree_elbow_thresholding(
            echidna.eta_ground_truth,
            plot_dendrogram=not elbow,
            plot_elbow=elbow,
        )
    elif method=="cophenetic":
        fig = eta_cov_tree_cophenetic_thresholding(
            echidna.eta_ground_truth,
            plot_dendrogram=True,
        )
    else:
        fig = eta_cov_tree(
            echidna.eta_ground_truth,
            thres=adata.uns["echidna"]["save_data"]["threshold"],
            plot_dendrogram=True,
        )
    if filepath: save_figure(fig, filepath)

def echidna(
    adata,
    color=["echidna_clones"],
    basis="X_umap",
    filepath=None,
    return_fig=False,
    **kwargs,
):
    """
    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    basis : str, default "X_umap"
        The basis to use for the plot.
    **kwargs : dict, optional
        Additional arguments passed to `sc.pl.embedding`.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The matplotlib figure.
    """
    activate_plot_settings()
    
    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        color_map="cool_r",
        frameon=True,
        show=True,
        sort_order=True,
        return_fig=True,
        **kwargs,
    )
    if filepath: save_figure(fig, filepath)
    if return_fig: return fig
