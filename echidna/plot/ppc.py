# echidna.plot.ppc.py

import matplotlib.pyplot as plt
import numpy as np

import torch
from pyro import render_model

from echidna.tools.housekeeping import load_model
from echidna.plot.utils import is_notebook

def plate_model(adata, filename: str=None):
    """Display plate model
    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix
    filename : str
        Saves figure to the given path.
    """
    echidna = load_model(adata)
    if echidna.config._is_multi:
        X_rand = torch.randint(low=0, high=40, size=(echidna.config.num_timepoints, 10, echidna.config.num_genes), dtype=torch.float32)
        W_rand = torch.rand((echidna.config.num_timepoints, echidna.config.num_genes), dtype=torch.float32)
        z_rand = torch.randint(low=0, high=echidna.config.num_clusters, size=(echidna.config.num_timepoints, 10,), dtype=torch.int32)
        pi_rand = torch.rand((echidna.config.num_timepoints, echidna.config.num_clusters), dtype=torch.float32)
    else:
        X_rand = torch.randint(low=0, high=40, size=(10, echidna.config.num_genes), dtype=torch.float32)
        W_rand = torch.rand((echidna.config.num_genes), dtype=torch.float32)
        z_rand = torch.randint(low=0, high=echidna.config.num_clusters, size=(10,), dtype=torch.int32)
        pi_rand = torch.rand((echidna.config.num_clusters), dtype=torch.float32)
    
    data=(X_rand, W_rand, pi_rand, z_rand)
    fig = render_model(
        echidna.model, 
        model_args=data, 
        render_params=True, 
        render_distributions=True,
        render_deterministic=True,
        filename=filename,
    )
    if is_notebook():
        display(fig)

def ppc(adata, variable):
    # Given a vairable, view its ppc
    pass

def plot_true_vs_pred(
    X_learned: np.ndarray,
    X_true: np.ndarray,
    name: str = "",
    log_scale: bool = False,
    save: bool = True,
    color: str = None,
):
    """
    Plot X learned vs. True
    """
    # Subsample for plotting
    num = min(len(X_learned.flatten()), 200000)
    indx = np.random.choice(np.arange(len(X_learned.flatten())), num, replace=False)
    X_learned = X_learned.flatten()[indx]
    X_true = X_true.flatten()[indx]

    if log_scale:
        X_learned = np.log(X_learned + 1)
        X_true = np.log(X_true + 1)
        lbl_pstfix = "[log(x + 1)] "
    else:
        lbl_pstfix = ""

    x = X_true
    y = X_learned

    maximum = max(np.max(x), np.max(y))
    minimum = min(np.min(x), np.min(y))
    # scatter plot
    plt.scatter(x, y, alpha=0.1)
    plt.plot([minimum, maximum], [minimum, maximum], "r", label="x=y")
    plt.xlabel("True " + lbl_pstfix)
    plt.ylabel("Learned" + lbl_pstfix)

    # Fit a line through x and y
    color = 'g--' if color is None else color + '--'
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), color, label="Fit line")
    plt.title(f"Comprarison of true and learned vals {name} (subsampled)")
    plt.legend()