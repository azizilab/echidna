# echidna.plot.ppc.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

import torch
from pyro import render_model

from echidna.tools.housekeeping import load_model, get_learned_params
from echidna.tools.data import build_torch_tensors
from echidna.tools.eval import sample
from echidna.tools.utils import EchidnaConfig, _custom_sort
from echidna.plot.utils import is_notebook, activate_plot_settings
from scipy.stats import linregress, gaussian_kde


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
    data = build_torch_tensors(adata_tmp, config)
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
    pred_posterior_check(
        X_learned, X_true, name='X', log_scale=True, R_val=False,
        color_by_density=False, title="Observed vs. reconstructed",
        xlabname="True ", ylabname="Reconstructed ", filename=filename,
    )
    
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
    
    pred_posterior_check(W_learned, W_true, name="W", filename=filename)

def ppc_cov(
    adata,
    learned_params,
    difference: bool=False,
    corr: bool=False,
    filename: str=None,
    cmap: str='plasma_r'
):
    echidna_sim = load_model(adata, simulation=True)
    echidna = load_model(adata)
    
    # Calculate the difference matrix
    cov_matrix_simulated = echidna_sim.cov_posterior
    cov_matrix_real = echidna.cov_posterior
    cov_matrix_diff = cov_matrix_real - cov_matrix_simulated
    
    if corr is True:
        sim_std = torch.sqrt(torch.diag(cov_matrix_simulated))
        real_std = torch.sqrt(torch.diag(cov_matrix_real))
        
        cov_matrix_simulated = cov_matrix_simulated / torch.outer(sim_std, sim_std)
        cov_matrix_real = cov_matrix_real / torch.outer(real_std, real_std)
        
        cov_matrix_simulated[torch.isnan(cov_matrix_simulated)] = 0
        cov_matrix_real[torch.isnan(cov_matrix_real)] = 0
        
        cov_matrix_diff = cov_matrix_real - cov_matrix_simulated
    
    cov_matrix_diff = cov_matrix_diff.detach().cpu().numpy()
    cov_matrix_simulated = cov_matrix_simulated.detach().cpu().numpy()
    cov_matrix_real = cov_matrix_real.detach().cpu().numpy()
    cov_corr_title = "Correlation" if corr is True else "Covariance"
    
    if difference is True:
        # Create heatmaps
        plt.figure(figsize=(18, 6))
        
        # Plot the real covariance matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cov_matrix_real, cmap=cmap, annot=True, fmt='.2f')
        plt.title(f'{cov_corr_title} Matrix (Real)')
    
        # Plot the simulated covariance matrix
        plt.subplot(1, 3, 2)
        sns.heatmap(cov_matrix_simulated, cmap=cmap, annot=True, fmt='.2f')
        plt.title(f'{cov_corr_title} Matrix (Learned)')
    
        # Plot the difference between the two covariance matrices
        plt.subplot(1, 3, 3)
        sns.heatmap(cov_matrix_diff, cmap=cmap, annot=True, fmt='.2f')
        plt.title('Difference (Real - Learned)')
        plt.tight_layout()
    elif difference is False:
        n = cov_matrix_real.shape[-1]
        df1 = pd.DataFrame(cov_matrix_simulated, columns=[f"Clst {i}" for i in range(n)], index=[f"Clst {i}" for i in range(n)])
        df2 = pd.DataFrame(cov_matrix_real, columns=[f"Clst {i}" for i in range(n)], index=[f"Clst {i}" for i in range(n)])
        
        # Perform hierarchical clustering on the first dataset
        linkage_rows = linkage(df1, method='average', metric='euclidean')
        linkage_cols = linkage(df1.T, method='average', metric='euclidean')
    
        # Get the order of the rows and columns
        row_order = leaves_list(linkage_rows)
        col_order = leaves_list(linkage_cols)
    
        # Reorder both datasets
        df1_ordered = df1.iloc[row_order, col_order]
        df2_ordered = df2.iloc[row_order, col_order]
    
        # Create a grid for the plots
        fig = plt.figure(figsize=(20, 10))
    
        # Define the axes for the first plot
        gs = fig.add_gridspec(3, 4, width_ratios=[0.05, 1, 0.05, 1], height_ratios=[0.2, 1, 0.05], wspace=0.1, hspace=0.1)
        ax_col_dendro1 = fig.add_subplot(gs[0, 1])
        ax_heatmap1 = fig.add_subplot(gs[1, 1])
    
        # Define the axes for the second plot
        ax_col_dendro2 = fig.add_subplot(gs[0, 3])
        ax_heatmap2 = fig.add_subplot(gs[1, 3])
    
        # Plot dendrogram for columns of the first dataset
        dendro_col1 = dendrogram(linkage_cols, ax=ax_col_dendro1, orientation='top', no_labels=True, color_threshold=0)
        ax_col_dendro1.set_xticks([])
        ax_col_dendro1.set_yticks([])
        ax_col_dendro1.set_title(f"Refitted {cov_corr_title}")
    
        # Plot heatmap for the first dataset
        sns.heatmap(df1_ordered, ax=ax_heatmap1, cmap=cmap, cbar=False, xticklabels=False, yticklabels=True)
    
        # Plot dendrogram for columns of the second dataset
        dendro_col2 = dendrogram(linkage_cols, ax=ax_col_dendro2, orientation='top', no_labels=True, color_threshold=0)
        ax_col_dendro2.set_xticks([])
        ax_col_dendro2.set_yticks([])
        ax_col_dendro2.set_title(f"Original {cov_corr_title}")
    
        # Plot heatmap for the second dataset
        sns.heatmap(df2_ordered, ax=ax_heatmap2, cmap=cmap, cbar=False, xticklabels=False, yticklabels=True)
        
    if filename: plt.savefig(filename)
    plt.show()

def ppc_c(adata, learned_params, filename: str=None):
    
    echidna = load_model(adata)
    
    c_learned = learned_params["c"]["value"].flatten().detach().cpu().numpy()
    c_posterior = echidna.c_posterior.flatten().detach().cpu().numpy()
    c_learned, c_posterior = _sample_arrays(c_learned, c_posterior, seed=echidna.config.seed)
    
    del echidna
    
    slope, intercept, r_value, p_value, std_err = linregress(c_learned, c_posterior)
    r_squared = r_value

    data = pd.DataFrame({'c_learned': c_learned, 'c_posterior': c_posterior})

    pred_posterior_check(c_learned, c_posterior, "c", equal_line=False, title="Refitted vs. posterior ground truth ",
                         xlabname="Refitted ", ylabname="Posterior ground truth ", filename=filename)

    #plt.figure(figsize=(10, 6))
    #regplot = sns.regplot(data=data, x='c_learned', y='c_posterior', scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'red'})
    #scatter = sns.scatterplot(data=data, x='c_learned', y='c_posterior', s=50, color='blue', alpha=0.6)

    #plt.text(0.05, 0.95, f'$Pearson R = {r_squared:.4f}$', transform=plt.gca().transAxes,
             #fontsize=12, verticalalignment='top')

    #plt.title('c fitted vs. c posterior truth')
    #plt.xlabel('c fitted')
    #plt.ylabel('c posterior truth')
    
    #if filename: plt.savefig(filename)
    #plt.show()


def ppc_eta(adata, learned_params, filename: str=None):
    echidna_sim = load_model(adata, simulation=True)
    echidna = load_model(adata)
    
    eta_learned = echidna_sim.eta_posterior.flatten().detach().cpu().numpy()
    eta_posterior = echidna.eta_posterior.flatten().detach().cpu().numpy()
    
    eta_learned, eta_posterior = _sample_arrays(eta_learned, eta_posterior, seed=echidna.config.seed)
    
    #slope, intercept, r_value, p_value, std_err = linregress(eta_learned, eta_posterior)
    #r_squared = r_value

    data = pd.DataFrame({'eta_learned': eta_learned, 'eta_posterior': eta_posterior})

    pred_posterior_check(eta_learned, eta_posterior, "$\eta$", equal_line=False, title="Refitted vs. posterior ground truth ",
                         xlabname="Refitted ", ylabname="Posterior ground truth ", filename=filename)

    #scatter = sns.scatterplot(data=data, x='eta_learned', y='eta_posterior', s=50, color='blue', alpha=0.6)
    #contour = sns.kdeplot(data=data, x='eta_learned', y='eta_posterior', levels=10, color='red', linewidths=1.5)

    #for child in scatter.get_children():
     #if isinstance(child, plt.Line2D): 
        #child.set_rasterized(True)

    #plt.text(
        #0.05, 0.95,
        #f'$Pearson R = {r_squared:.4f}$',
        #transform=scatter.transAxes,
        #fontsize=12,
        #verticalalignment='top',
    #)

    #scatter.set_title('eta fitted vs. eta posterior truth')
    #scatter.set_xlabel('eta fitted')
    #scatter.set_ylabel('eta posterior truth')

    #if filename: plt.savefig(filename)
    #plt.show()
    
    #del echidna
    #del echidna_sim

def _sample_arrays(learned, observed, max_samples=int(3e4), seed=42):
    rng = np.random.default_rng(seed)
    total_samples = min(len(learned), max_samples)
    indices = np.random.choice(len(learned), total_samples, replace=False)
    
    learned = learned[indices]
    observed = observed[indices]
    
    return learned, observed    

def pred_posterior_check(
        X_learned: np.ndarray,
        X_true: np.ndarray,
        name: str = "",
        log_scale: bool = False,
        R_val: bool = True,
        equal_line: bool = True,
        color_by_density: bool = False,
        title: str = "Predictive Posterior Check",
        xlabname: str = "True ",
        ylabname: str = "Simulated ",
        filename: str = None,
        subsample: bool = True
):
    if subsample:
      # Subsample for plotting
      num = min(len(X_learned.flatten()), 200000)
      indx = np.random.choice(
          np.arange(len(X_learned.flatten())), num, replace=False)
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

    # Perform linear regression
    if R_val:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value**2
        y_pred = slope * x + intercept

    maximum = max(np.max(x), np.max(y))
    minimum = min(np.min(x), np.min(y))

    # Scatter plot
    if color_by_density:
        # Calculate point densities
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Plot using density as color
        plt.scatter(x, y, c=z, cmap='viridis',
                    alpha=0.5, label='Data points', s=10, rasterized=True)
    else:
        plt.scatter(x, y, alpha=0.6, c='blue',  rasterized=True)

    if equal_line:
        plt.plot([minimum, maximum], [minimum, maximum], "r", label="x=y")

    # Plot the regression line
    #if R_val:
        #plt.plot(x, y_pred, label="Regression line", color='blue')

    plt.xlabel(xlabname + lbl_pstfix + name, fontsize=16)
    plt.ylabel(ylabname + lbl_pstfix + name, fontsize=16)

    # Annotate the plot with the R^2 value
    if R_val:
        plt.text(0.05, 0.95, f'$PearsonR$ = {r2:.3f}', ha='left', va='center', transform=plt.gca(
        ).transAxes, fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc="lower right")
    plt.title(title + " " + name, fontsize = 16)

    if filename is not None:
        plt.savefig(filename, format='svg')

    plt.show()
