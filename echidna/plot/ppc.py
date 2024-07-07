# echidna.plot.ppc.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress

import torch
from pyro import render_model

from echidna.tools.housekeeping import load_model, get_learned_params
from echidna.tools.data import convert_torch
from echidna.tools.eval import sample
from echidna.tools.utils import EchidnaConfig, _custom_sort
from echidna.plot.utils import is_notebook, activate_plot_settings

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

def ppc(adata, variable, **kwargs):
    if variable not in ("X", "W", "c", "eta", "cov"):
        raise ValueError(
            "`variable` must be one of or a list of "
            "(\"X\", \"W\", \"c\", \"eta\", \"cov\")"
        )
    activate_plot_settings()
    
    adata_tmp = adata[adata.obs["echidna_split"] != "discard", adata.var.echidna_matched_genes].copy()
    config = EchidnaConfig.from_dict(adata_tmp.uns["echidna"]["config"])
    data = convert_torch(adata_tmp, config)
    learned_params = get_learned_params(load_model(adata_tmp), data)

    ppc_funcs = {
        "X" : ppc_X,
        "W" : ppc_W,
        "c" : ppc_c,
        "eta" : ppc_eta,
        "cov" : ppc_cov,
    }
    
    return ppc_funcs[variable](adata, learned_params, **kwargs)

def ppc_X(adata, learned_params, filename: str=None):
    config = EchidnaConfig.from_dict(adata.uns["echidna"]["config"])
    cell_filter = adata.obs["echidna_split"] != "discard"
    gene_filter = adata.var["echidna_matched_genes"]
    adata_tmp = adata[cell_filter, gene_filter].copy()
    
    # X_true = []
    # if config._is_multi:
    #     timepoints = np.unique(adata_tmp.obs[config.timepoint_label])
    #     if "timepoint_order" in adata_tmp.uns["echidna"]:
    #         timepoints = _custom_sort(
    #             timepoints,
    #             adata_tmp.uns["echidna"]["timepoint_order"]
    #         )
    #     for tp in timepoints:
    #         tp_filter = adata_tmp.obs[config.timepoint_label] == tp
    #         X_true.append(adata_tmp[tp_filter].layers[config.counts_layer])
    #     X_true = np.array(X_true)
    # else:
    #     X_true = adata_tmp.layers[config.counts_layer]

    X_true = []
    for k in learned_params:
        if "X" in k: X_true.append(
            learned_params[k]["value"].detach().cpu().numpy()
        )
    X_true = np.array(X_true).squeeze()
    
    X_learned = sample(adata_tmp, "X").detach().cpu().numpy()

    plot_true_vs_pred(X_learned, X_true, log_scale=True, name="X", filename=filename)
    
def ppc_W(adata, learned_params, filename: str=None):
    # config = EchidnaConfig.from_dict(adata.uns["echidna"]["config"])
    # Wdf = adata.var[[c for c in adata.var.columns if "echidna_W_" in c]].dropna()
    # if config._is_multi and "timepoint_order" in adata.uns["echidna"]:
    #     Wdf = _custom_sort(Wdf, adata.uns["echidna"]["timepoint_order"])
    #     W_true = Wdf.values.T
    # else:
    #     W_true = Wdf.values
    
    # In learned_params, W and X were observed.
    W_true = learned_params["W"]["value"].detach().cpu().numpy()
    W_learned = sample(
        adata[:, adata.var.echidna_matched_genes].copy(), "W"
    ).detach().cpu().numpy()
    
    plot_true_vs_pred(W_learned, W_true, name="W", filename=filename)

def ppc_cov(adata, learned_params, filename: str=None):
    echidna_sim = load_model(adata, simulation=True)
    echidna = load_model(adata)
    
    # Calculate the difference matrix
    cov_matrix_simulated = echidna_sim.cov_ground_truth.detach().cpu().numpy()
    cov_matrix_real = echidna.cov_ground_truth.detach().cpu().numpy()
    cov_matrix_diff = cov_matrix_real - cov_matrix_simulated

    # Create heatmaps
    plt.figure(figsize=(18, 6))

    # Plot the real covariance matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cov_matrix_real, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Covariance Matrix (Real)')

    # Plot the simulated covariance matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(cov_matrix_simulated, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Covariance Matrix (Learned)')

    # Plot the difference between the two covariance matrices
    plt.subplot(1, 3, 3)
    sns.heatmap(cov_matrix_diff, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Difference (Real - Learned)')
    plt.tight_layout()
    
    if filename: plt.savefig(filename)
    plt.show()

def ppc_c(adata, learned_params, filename: str=None):
    
    echidna = load_model(adata)
    
    c_learned = learned_params["c"]["value"].flatten().detach().cpu().numpy()
    c_ground_truth = echidna.c_ground_truth.flatten().detach().cpu().numpy()
    c_learned, c_ground_truth = _sample_arrays(c_learned, c_ground_truth, seed=echidna.config.seed)
    
    del echidna
    
    slope, intercept, r_value, p_value, std_err = linregress(c_learned, c_ground_truth)
    r_squared = r_value**2

    data = pd.DataFrame({'c_learned': c_learned, 'c_ground': c_ground_truth})

    plt.figure(figsize=(10, 6))
    regplot = sns.regplot(data=data, x='c_learned', y='c_ground', scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'red'})
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.title('c fitted vs. c ground truth')
    plt.xlabel('c fitted')
    plt.ylabel('c ground truth')
    
    if filename: plt.savefig(filename)
    plt.show()

def ppc_eta(adata, learned_params, filename: str=None):
    echidna_sim = load_model(adata, simulation=True)
    echidna = load_model(adata)
    
    eta_learned = echidna_sim.eta_ground_truth.flatten().detach().cpu().numpy()
    eta_ground = echidna.eta_ground_truth.flatten().detach().cpu().numpy()
    
    eta_learned, eta_ground = _sample_arrays(eta_learned, eta_ground, seed=echidna.config.seed)
    
    slope, intercept, r_value, p_value, std_err = linregress(eta_learned, eta_ground)
    r_squared = r_value**2

    data = pd.DataFrame({'eta_learned': eta_learned, 'eta_ground': eta_ground})

    scatter = sns.scatterplot(data=data, x='eta_learned', y='eta_ground', s=50, color='blue', alpha=0.6)
    contour = sns.kdeplot(data=data, x='eta_learned', y='eta_ground', levels=10, color='red', linewidths=1.5)

    plt.text(
        0.05, 0.95,
        f'$R^2 = {r_squared:.4f}$',
        transform=scatter.transAxes,
        fontsize=12,
        verticalalignment='top',
    )

    scatter.set_title('eta fitted vs. eta ground truth')
    scatter.set_xlabel('eta fitted')
    scatter.set_ylabel('eta ground truth')

    if filename: plt.savefig(filename)
    plt.show()
    
    del echidna
    del echidna_sim

def _sample_arrays(learned, observed, max_samples=int(3e4), seed=42):
    rng = np.random.default_rng(seed)
    total_samples = min(len(learned), max_samples)
    indices = np.random.choice(len(learned), total_samples, replace=False)
    
    learned = learned[indices]
    observed = observed[indices]
    
    return learned, observed    

def plot_true_vs_pred(
    X_learned: np.ndarray,
    X_true: np.ndarray,
    name: str="",
    log_scale: bool=False,
    color: str=None,
    filename: str=None,
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
    if filename: plt.savefig(filename)
    plt.show()
