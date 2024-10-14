# echidna.tools.post.py

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np
import pandas as pd
import os
from typing import Union, List
import torch

from echidna.tools.eval import (
    eta_tree,
    eta_tree_elbow_thresholding,
    eta_tree_cophenetic_thresholding,
)
from echidna.tools.infer_cnv import gene_dosage_effect
from echidna.tools.housekeeping import load_model
from echidna.tools.data import sort_chromosomes, filter_low_var_genes
from echidna.plot.utils import save_figure, activate_plot_settings
from echidna.utils import get_logger

logger = get_logger(__name__)

def plot_cnv(adata, c: str=None, filename: str=None):
    c = "all" if c is None else c
    
    # Retrieve save data from tools functions
    if "infer_cnv" not in adata.uns["echidna"]["save_data"]:
        raise ValueError("Must run `ec.tl.infer_cnv` first.")
    file_save_path = adata.uns["echidna"]["save_data"]["infer_cnv"]
    neutral_save_path = adata.uns["echidna"]["save_data"]["gmm_neutrals"]
    if not os.path.exists(file_save_path):
        raise ValueError(
            "Saved results not found. Run `ec.tl.infer_cnv` first."
        )
    eta_genome_merge = pd.read_csv(file_save_path)
    neutral_states = pd.read_csv(
        neutral_save_path,
        index_col="eta_column_label",
    )
    num_clusters = adata.obs[adata.uns["echidna"]["config"]["clusters"]].nunique()
    
    # band_means_states["chrom"] = band_means_states["band"].str.extract(r"^(chr[0-9XY]+)_")[0]
    
    for i in [f"echidna_clone_{i}" for i in range(num_clusters)]:
        eta_genome_merge[i] -= neutral_states.loc[i, "neutral_value_mean"].item()
    
    chrom_counts = sort_chromosomes(
        eta_genome_merge.groupby("chrom")["geneName"].nunique()
    ).cumsum()
    
    activate_plot_settings()
    if c != "all":
        if f"echidna_clone_{c}" not in eta_genome_merge.columns:
            raise ValueError(f"Specified cluster `echidna_clone_{c}` not found.")
        vals = eta_genome_merge.loc[:, f"echidna_clone_{c}"]
        states = eta_genome_merge.loc[:, f"states_echidna_clone_{c}"]
        _plot_cnv_helper(
            vals,
            states,
            chrom_counts.values,
            chrom_counts.index,
            title=f"Echidna Clone {c} CNV",
            filename=filename,
        )  
    elif c == "all":
        fig, ax = plt.subplots(figsize=(20,5), nrows=1, ncols=1)
        
        eta_genome_merge = eta_genome_merge[
            [f"echidna_clone_{i}" for i in range(num_clusters)]
        ]
        eta_genome_merge.columns = [f"{i}" for i in range(num_clusters)]
        sns.heatmap(eta_genome_merge.T, cmap="bwr", ax=ax, vmin=-2, vmax=2)
        
        # Set the x-axis ticks and labels
        ticks = [(chrom_counts[i-1] + chrom_counts[i])/2 if i != 0 else chrom_counts[i]/2 for i in range(len(chrom_counts))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(chrom_counts.index, rotation=30)

        # Draw vertical lines at each chromosome boundary
        for x in chrom_counts:
            ax.axvline(x=x, color="k", linestyle="--", linewidth=1.5)
         
        # fig, axes = plt.subplots(num_clusters, 1, figsize=(25, 7 * num_clusters))

        # for i in range(num_clusters):
        #     vals = band_means_states.loc[:, f"echidna_clone_{i}"]
        #     states = band_means_states.loc[:, f"states_echidna_clone_{i}"]
        #     _plot_cnv_helper(
        #         vals,
        #         states,
        #         chrom_counts.values,
        #         chrom_counts.index,
        #         ax=axes[i],
        #         title=f"Echidna Clone {i} CNV",
        #         filename=None,
        #     )
        
        if filename: fig.savefig(filename, format="png")

def plot_gene_dosage(
    adata,
    clusters: Union[int, List[int]]=None,
    timepoints: Union[int, List[int]]=None,
    quantile: float=.75,
    var_threshold: float=None,
    filename: str=None,
) -> None:
    model = load_model(adata)
    
    if "gene_dosage" not in adata.uns["echidna"]["save_data"]:
        gene_dosage = gene_dosage_effect(adata)
    else:
        gene_dosage_cache = adata.uns["echidna"]["save_data"]["gene_dosage"]
        if not os.path.exists(gene_dosage_cache):
            raise ValueError(
                "Saved results not found. Run `ec.tl.gene_dosage_effect` first."
            )
        gene_dosage = torch.load(gene_dosage_cache).to(model.config.device)
    
    # Ensure clusters is a list of integers
    if clusters is not None:
        if isinstance(clusters, int):
            clusters = [clusters]
        elif not all(isinstance(cl, int) for cl in clusters):
            raise ValueError("Clusters must be an integer or a list of integers.")
        
        if not all(0 <= cl < gene_dosage.shape[-1] for cl in clusters):
            raise ValueError(
                f"Invalid cluster values: {clusters}. "
                f"Clusters must be integers in the range [0, {gene_dosage.shape[-1] - 1}]."
            )
    else:
        clusters = list(range(0, gene_dosage.shape[-1]))

    # Ensure timepoints is a list of integers
    if timepoints is not None:
        if isinstance(timepoints, int):
            timepoints = [timepoints]
        elif not all(isinstance(tp, int) for tp in timepoints):
            raise ValueError("Timepoints must be an integer or a list of integers.")
        
        if not all(0 <= tp < gene_dosage.shape[0] for tp in timepoints):
            raise ValueError(
                f"Invalid timepoint values: {timepoints}. "
                f"Timepoints must be integers in the range [0, {gene_dosage.shape[0] - 1}]."
            )
    else:
        timepoints = list(range(0, gene_dosage.shape[0]))
    
    neutral_save_path = adata.uns["echidna"]["save_data"]["gmm_neutrals"]
    neutral_states = pd.read_csv(
        neutral_save_path,
        index_col="eta_column_label",
    )
    eta = model.eta_ground_truth
    echidna_matched_genes = adata[
        :, adata.var["echidna_matched_genes"]
    ].var.index

    eta = pd.DataFrame(
        model.eta_ground_truth.T.cpu().detach().numpy(),
        index=echidna_matched_genes,
        # columns=[f"echidna_clone_{i}" for i in range(num_clusters)]
    )
    
    filter_genes = filter_low_var_genes(
        adata[:, adata.var["echidna_matched_genes"]].copy(),
        quantile=quantile,
        var_threshold=var_threshold,
        indices=True,
    )

    filter_genes_tensor = torch.tensor(filter_genes, device=model.config.device)
    filter_genes_indices = torch.nonzero(filter_genes_tensor, as_tuple=False).squeeze()
    eta = eta[filter_genes]
    
    n_rows = len(clusters)
    n_cols = len(timepoints)
    
    # Set the figsize dynamically based on the number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    
    # Loop over each cluster and timepoint to create subplots
    for i, tp in enumerate(timepoints):
        gene_dosage_df = pd.DataFrame(
                gene_dosage[tp, filter_genes_indices, :].abs().cpu().detach().numpy(),
                index=eta.index,#adata[:, adata.var["echidna_matched_genes"]].var.index,
        )
        for j, cl in enumerate(clusters):

            ax = axes[j][i]
            
            sns.scatterplot(
                x=eta[cl] - neutral_states.loc[f"echidna_clone_{cl}", "neutral_value_mean"],
                y=gene_dosage_df[cl],
                ax=ax
            )
            ax.set_title(f"Cluster {cl}, Timepoint {tp}")
            ax.set_xlabel("Eta Shifted")
            ax.set_ylabel("Gene Dosage Effect")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
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
    
    # for i, row in df.iterrows():
        # ax.axvline(x=row["x"], color=row["color"], linestyle="-", alpha=0.3, linewidth=1)
    
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
