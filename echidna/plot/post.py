# echidna.tools.post.py

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
import os

from echidna.tools.eval import (
    eta_tree,
    eta_tree_elbow_thresholding,
    eta_tree_cophenetic_thresholding,
)
from echidna.tools.housekeeping import load_model
from echidna.tools.data import sort_chromosomes
from echidna.plot.utils import save_figure, activate_plot_settings
from echidna.utils import get_logger

logger = get_logger(__name__)

def plot_cnv(adata, c: str=None, filename: str=None):
    c = "all" if c is None else c
    
    if "infer_cnv" not in adata.uns["echidna"]["save_data"]:
        raise ValueError("Must run `ec.tl.infer_cnv` first.")
    file_save_path = adata.uns["echidna"]["save_data"]["infer_cnv"]
    if not os.path.exists(file_save_path):
        raise ValueError(
            "Saved results not found. Run `ec.tl.infer_cnv` first."
        )
    
    band_means_states = pd.read_csv(file_save_path)
    
    band_means_states["chrom"] = band_means_states["band"].str.extract(r"^(chr[0-9XY]+)_")[0]
    chrom_counts = sort_chromosomes(
        band_means_states.groupby("chrom")["band"].nunique()
    ).cumsum()
    
    activate_plot_settings()
    if c != "all":
        if f"echidna_clone_{c}" not in band_means_states.columns:
            raise ValueError(f"Specified cluster `{c}` not found.")
        vals = band_means_states.loc[:, f"echidna_clone_{c}"]
        states = band_means_states.loc[:, f"states_echidna_clone_{c}"]
        _plot_cnv_helper(
            vals,
            states,
            chrom_counts.values,
            chrom_counts.index,
            title=f"Echidna Clone {c} CNV",
            filename=filename,
        )  
    elif c == "all":
        num_clusters = adata.obs[adata.uns["echidna"]["config"]["clusters"]].nunique()
        fig, axes = plt.subplots(num_clusters, 1, figsize=(25, 7 * num_clusters))

        for i in range(num_clusters):
            vals = band_means_states.loc[:, f"echidna_clone_{i}"]
            states = band_means_states.loc[:, f"states_echidna_clone_{i}"]
            _plot_cnv_helper(
                vals,
                states,
                chrom_counts.values,
                chrom_counts.index,
                ax=axes[i],
                title=f"Echidna Clone {i} CNV",
                filename=None,
            )
        if filename: fig.savefig(filename, format="png")

def plot_gmm_clusters(gmm, vals_filtered, gmm_mean, i):
    fig, ax = plt.subplots(
            nrows=1,
            ncols=1, 
            figsize=(10, 5)
    )

    x = np.linspace(-5, 10, 1000)
    data = np.asarray(vals_filtered).squeeze()

    sns.histplot(data, bins=30, stat='density', alpha=0.6, color='gray', ax=ax)

    for mean, variance, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        variance = np.sqrt(variance).flatten()
        label = f'Component mean={mean[0]:.3f}'
        color = 'red' if np.isclose(mean[0], gmm_mean, atol=0.1) else None
        linewidth = 3 if color == 'red' else 1

        ax.plot(
            x,
            weight * (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance)),
            label=f'Neutral mean = {gmm_mean:.3f}' if color else label,
            color=color,
            linewidth=linewidth
        )

    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    ax.plot(x, pdf, '-k', label='Global Density')

    ax.set_xlabel('Eta values')
    ax.set_ylabel('Density')
    ax.set_title(f"Echidna Cluster {i}")
    ax.legend()

    plt.tight_layout()
    plt.show()

def _plot_cnv_helper(vals, states, chrom_coords, chroms, ax=None, title=None, filename=None):
    """Plot the CNV states along the genome.

    Parameters
    ----------
        vals : list/np.ndarray
            List of ordered copy number values (from bin_by_bands function)
        states : list/np.ndarray
            List of CN state calls from the HMM (get_states function)
        chrom_coords : list
            list of coordinates of the end of each chromosome (from bin_by_bands)
        chroms : list
            Ordered unique list of chromosome names
        title : str (optional)
            Title to label the plot
        filename : str (optional)
            Name of the file to save the plot
    """
    df = pd.DataFrame({
        "x": np.arange(len(vals)),
        "vals": vals,
        "states": states,
    })
    
    color_map = {"neut": "grey", "amp": "red", "del": "blue"}
    df["color"] = df["states"].map(color_map)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 5))
    
    for i, row in df.iterrows():
        ax.axvline(x=row["x"], color=row["color"], linestyle="-", alpha=0.3, linewidth=1)
    
    sns.scatterplot(x="x", y="vals", hue="states", palette=color_map, data=df, legend=False, s=80, ax=ax)
    
    # Set the x-axis ticks and labels
    ticks = [(chrom_coords[i-1] + chrom_coords[i])/2 if i != 0 else chrom_coords[i]/2 for i in range(len(chrom_coords))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(chroms, rotation=30)
    
    # Draw vertical lines at each chromosome boundary
    for x in chrom_coords:
        ax.axvline(x=x, color="k", linestyle="--", linewidth=1.2)
        
    ax.set_xlabel("Bands")
    ax.set_ylabel("CN")
    
    ax.grid(axis="x")
    
    if title:
        ax.set_title(title)
    if filename:
        ax.figure.savefig(filename, format="png")
    
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
        metric = adata.uns["echidna"]["save_data"]["dendrogram_metric"]
    except Exception as e:
        logger.error(f"Must run `ec.tl.echidna_clones` first. {e}")
        return

    if elbow and method != "elbow":
        logger.warning("`elbow=True` only applies to `method=\"elbow\"`.")

    if method == "elbow":
        fig = eta_tree_elbow_thresholding(
            echidna.eta_ground_truth,
            similarity_metric=metric,
            plot_dendrogram=not elbow,
            plot_elbow=elbow,
        )
    elif method == "cophenetic":
        fig = eta_tree_cophenetic_thresholding(
            echidna.eta_ground_truth,
            similarity_metric=metric,
            plot_dendrogram=True,
        )
    else:
        fig = eta_tree(
            echidna.eta_ground_truth,
            similarity_metric=metric,
            thres=adata.uns["echidna"]["save_data"]["threshold"],
            plot_dendrogram=True,
        )

    if filepath: save_figure(fig, filepath)
    del echidna

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
