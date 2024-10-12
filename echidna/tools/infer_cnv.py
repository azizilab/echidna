# echidna.tools.hmm.py

import logging, os, re, sys

import scanpy as sc
import numpy as np
import pandas as pd
import torch
import pyro

from scipy.stats import ttest_ind, mode
from scipy.ndimage import gaussian_filter1d
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture

from echidna.tools.housekeeping import load_model
from echidna.tools.data import filter_low_var_genes, sort_chromosomes
from echidna.tools.eval import sample
from echidna.utils import get_logger, ECHIDNA_GLOBALS
from echidna.plot.post import plot_gmm_clusters

logger = get_logger(__name__)

def cnv_results(adata: sc.AnnData) -> pd.DataFrame:
    infer_cnv_save_path = None
    if "infer_cnv" not in adata.uns["echidna"]["save_data"]:
        raise ValueError(
            "Must run `ec.tl.infer_cnv` first."
        )
    infer_cnv_save_path = adata.uns["echidna"]["save_data"]["infer_cnv"]
    if not os.path.exists(infer_cnv_save_path):
        raise ValueError(
            "Saved results not found. Run `ec.tl.infer_cnv` first."
        )
    
    return pd.read_csv(infer_cnv_save_path)

def infer_cnv(
    adata: sc.AnnData,
    genome: pd.DataFrame=None,
    gaussian_smoothing: bool=True,
    filter_genes: bool=True,
    filter_quantile: float=.7,
    smoother_sigma: float=6,
    smoother_radius: float=8,
    **kwargs,
) -> None:
    if genome is None:
        try:
            genome = pd.read_csv(
                "https://web.cs.ucla.edu/~wob/data/GRCh38_cytoband_gencodeV46.csv"
            )
            logger.info(
                "`genome` not set, defaulting to hg38"
                "cytoBands and wgEncodeGencodeCompV46."
            )
        except Exception as e:
            logger.info(
                "`genome` not set, defaulting to hg38"
                "cytoBands and wgEncodeGencodeCompV46."
            )
            logger.error(e)
            raise IOError("Must enable internet connection to fetch default genome data.")
    
    if "chr" not in genome["band"].iloc[0]:
        genome["band"] = genome["chrom"] + "_" + genome["band"]
    genome = sort_chromosomes(genome)
    
    echidna = load_model(adata)
    eta = echidna.eta_ground_truth
    
    del echidna
    torch.cuda.empty_cache()
    
    echidna_matched_genes = adata[:, adata.var["echidna_matched_genes"]].var.index
    
    eta = eta.T.detach().cpu().numpy()
    eta = pd.DataFrame(eta).set_index(echidna_matched_genes)
    clone_cols = [f"echidna_clone_{c}" for c in range(eta.shape[1])]
    eta.columns = clone_cols
    
    adata_filtered = filter_low_var_genes(adata.copy(), quantile=filter_quantile)
    common_genes = adata_filtered.var.index.intersection(
        eta.index
    ).intersection(genome.loc[:, "geneName"])
    genome_unique = genome.drop_duplicates(subset="geneName")
    eta_filtered = eta.reindex(
        genome_unique[genome_unique.loc[:, "geneName"].isin(common_genes)].loc[:, "geneName"]
    )
    
    # GMM gets eta filtered for genes and gaussian smoothed
    eta_filtered_smooth = gaussian_filter1d(
        eta_filtered, sigma=smoother_sigma, axis=0, radius=smoother_radius
    )
    neutral_states = get_neutral_state(eta_filtered_smooth, clone_cols, **kwargs)
    
    # User has choice for HMM on filtering genes and gaussian smoothing
    if filter_genes:
        eta = eta_filtered
    if gaussian_smoothing:
        eta.values[:] = gaussian_filter1d(
            eta, sigma=smoother_sigma, axis=0, radius=smoother_radius
        )

    # OLD - normalize out variance of each gene from eta
    # eta = eta / torch.sqrt(torch.diag(torch.cov(eta.T)))

    eta_genome_merge = genome[["geneName", "chrom"]].merge(
        eta, left_on="geneName",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    # if "weight" in genome.columns:
    #     bands_eta_merge[clone_cols] = bands_eta_merge[clone_cols].mul(
    #         bands_eta_merge["weight"], axis="index"
    #     )
    
    # bands_eta_means = bands_eta_merge.groupby("band")[clone_cols].mean()
    # bands_eta_means = bands_eta_means.reindex(
        # genome["band"].drop_duplicates()
    # ).dropna()
    
    for c in clone_cols:
        eta_genome_merge[f"states_{c}"] = _get_states(
            eta_genome_merge.loc[:, c].values,
            neutral_mean=neutral_states.loc[c, "neutral_value_mean"].item(),
            neutral_std=neutral_states.loc[c, "neutral_value_std"].item(),
            **kwargs
        )
        eta_genome_merge[c] -= neutral_states.loc[c, "neutral_value_mean"].item()

    infer_cnv_save_path = os.path.join(
        ECHIDNA_GLOBALS["save_folder"],
        adata.uns["echidna"]["run_id"],
        "_echidna_cnv.csv",
    )
    
    eta_genome_merge.to_csv(
        infer_cnv_save_path,
        index=False,
    )
    adata.uns["echidna"]["save_data"]["infer_cnv"] = infer_cnv_save_path
    logger.info(
        "Added `.uns['echidna']['save_data']['infer_cnv']` : Path to CNV inference results."
    )

def _get_states(
    vals,
    n_components=5,
    neutral_mean=2,
    neutral_std=1,
    transmat_prior=1,
    startprob_prior=1,
    verbose=False,
    **args,
) -> list:
    """Implements a Gaussian HMM to call copy number states smoothing along the genome
    
    Parameters
    ----------
    vals: ordered copy number values (from bin_by_bands function)
    n_components: number of components to use for the hmm. We default to 5 for better sensitivity.
    neutral: the number to use for neutral. 
    transmat_prior: Parameters of the Dirichlet prior distribution for each row of the transition probabilities
    startprob_prior: Parameters of the Dirichlet prior distribution for startprob_
    
    Returns
    -------
    pd.DataFrame
    """
    scores = []
    models = []

    idx = np.random.choice(range(100), size=10)
    for i in range(10):
        model = hmm.GaussianHMM(
            n_components=n_components,
            random_state=idx[i],
            n_iter=150,
            transmat_prior=transmat_prior,
            startprob_prior=startprob_prior,
        )
        model.fit(vals.reshape(-1, 1))
        models.append(model)
        scores.append(model.score(vals.reshape(-1, 1)))
        if verbose:
            logger.info(
                f"Converged: {model.monitor_.converged}\t\tScore: {scores[-1]}"
            )
    
    # Get the best model
    model = models[np.argmax(scores)]
    if verbose:
        logger.info(
            f"The best model had a score of {max(scores)}"
            "and {model.n_components} components"
        )
    
    states = model.predict(vals[:, None])

    tmp = pd.DataFrame({"vals": vals, "states": states})
    state_dict = tmp.groupby("states")["vals"].mean().reset_index()

    # state_dict["sq_dist_to_neut"] = (state_dict["vals"] - neutral)**2
    # state_dict["dist_to_neut"] = state_dict["vals"] - neutral
    # state_dict = state_dict.sort_values(by="sq_dist_to_neut")
    
    n_stddevs = 1.96
    state_dict["neutral"] = abs(state_dict["vals"] - neutral_mean) <= n_stddevs * neutral_std
    state_dict["amp"] = state_dict["vals"] - neutral_mean > n_stddevs * neutral_std
    state_dict["del"] = state_dict["vals"] - neutral_mean < -n_stddevs * neutral_std

    # Check if any other states are neutral using a t-test
    # neutral_state = [state_dict.iloc[0]["states"]]
    # neutral_val = [state_dict.iloc[0]["vals"]]
    # neutral_dist = tmp[tmp["states"] == neutral_state[0]]["vals"]

    # for i, row in state_dict.iterrows():
    #     if row["states"] not in neutral_state:
    #         tmp_dist = tmp[tmp["states"] == row["states"]]["vals"]
    #         p_val = ttest_ind(neutral_dist, tmp_dist).pvalue
    #         if p_val > p_thresh:
    #             neutral_state.append(row["states"])
    #             neutral_val.append(row["vals"])
    
    # amp if CN > netural, del if less than
    def classify_state(row):
        if row["neutral"] is True:
            return "neut"
        elif row["amp"] is True:
            return "amp"
        elif row["del"] is True:
            return "del"
    state_dict["CNV"] = state_dict.apply(classify_state, axis=1)
    
    # Map the CNV states back to the original states
    state_map = state_dict.set_index("states")["CNV"].to_dict()
    cnvs = list(map(state_map.get, states))

    return cnvs

def get_neutral_state(
    eta_filtered_smooth,
    eta_column_labels,
    n_gmm_components=5,
    plot_gmm=False,
    **args,
) -> pd.DataFrame:
    gmm_means_df = pd.DataFrame(columns=["eta_column_label", "mode", "neutral_value_mean", "neutral_value_std"])
    
    for i, col in enumerate(eta_column_labels):
        cur_vals_filtered = eta_filtered_smooth[:, i].reshape(-1,1)
        gmm = GaussianMixture(n_components=n_gmm_components).fit(cur_vals_filtered)
        labels = gmm.predict(cur_vals_filtered)
        neut_component = mode(labels, keepdims=False).mode

        gmm_mean = np.mean(cur_vals_filtered[labels == neut_component])
        gmm_std = np.std(cur_vals_filtered[labels == neut_component])

        if plot_gmm:
            plot_gmm_clusters(
                gmm, cur_vals_filtered, gmm_mean, i
            )

        gmm_means_df.loc[i, :] = [col, neut_component, gmm_mean, gmm_std]

    return gmm_means_df.set_index("eta_column_label")

def gene_dosage_effect(
    adata,
    smoother_sigma: float=6,
    smoother_radius: float=8,
    filter_quantile: float=.7,
    n_gmm_components: int=5,
    **kwargs
):
    """
    eta_mode [n_clusters, 1]
    eta_samples, eta_mean [n_genes, n_clusters]
    c_shape  [n_genes, n_timepoints]
    """
    model = load_model(adata)

    adata_filtered = filter_low_var_genes(
        adata.copy(), quantile=filter_quantile
    )
    
    echidna_matched_genes = adata[
        :, adata.var["echidna_matched_genes"]
    ].var.index
    
    eta = pd.DataFrame(
        model.eta_ground_truth.T.cpu().detach().numpy(),
        index=echidna_matched_genes,
    )
    echidna_matched_genes = echidna_matched_genes.intersection(adata_filtered.var.index)
    
    eta = eta.reindex(echidna_matched_genes)
    clone_cols = [f"echidna_clone_{c}" for c in range(eta.shape[1])]

    eta_filtered_smooth = gaussian_filter1d(
        eta, sigma=smoother_sigma, axis=0, radius=smoother_radius
    )
    eta_mode = get_neutral_state(
        eta_filtered_smooth, clone_cols, n_gmm_components
    )["neutral_value_mean"]
    
    eta_samples = sample(adata, "eta")
    c_shape = pyro.param("c_shape")
    eta_mean = pyro.param("eta_mean")
    eta_mode = torch.tensor(eta_mode, device=model.config.device)
    
    if adata.uns["echidna"]["config"]["_is_multi"]:
        c_shape = c_shape.squeeze()
        
    # delta Var(c|eta)
    delta_var = (c_shape[:, :, None] * eta_mean[None, :, :]**2) - c_shape[:, :, None] * eta_mode[None, :]
    
    # Var(E[c|eta])
    exp_c_given_eta = c_shape[:, :, None] * eta_samples[None, :, :]
    var_exp_c_given_eta = torch.var(exp_c_given_eta, unbiased=True, dim=1)
    
    # E[Var(c|eta)]
    exp_var_c_given_eta = c_shape[:, :, None] * eta_samples[None, :, :]**2
    exp_var_c_given_eta = torch.mean(exp_var_c_given_eta, dim=1)

    # (Var(c|eta)-Var(c|eta_mode)) / (Var(E[c|eta])+E[Var(c|eta)])
    var_exp = delta_var / (var_exp_c_given_eta + exp_var_c_given_eta)[:, None, :]
    
    return var_exp
    
def genes_to_bands(genes, cytobands):
    spanning_genes = []
    for i, gene in genes.iterrows():
        chrom = gene["chrom"]
        gene_start = gene["txStart"]
        gene_end = gene["txEnd"]
        
        lh_overlap = (
            (gene_start <=  cytobands["bandStart"]) &
            (gene_end >= cytobands["bandStart"])
        )
        in_between = (gene_start >= cytobands["bandStart"]) & (gene_end <= cytobands["bandEnd"])
        rh_overlap = (gene_start <= cytobands["bandEnd"]) & (gene_end >= cytobands["bandEnd"])
        
        bands = cytobands[(chrom == cytobands["chrom"]) & (lh_overlap | in_between | rh_overlap)].copy()
        if len(bands) > 0:
            span = gene_end - gene_start
            bands.loc[:, "weight"] = 1.
            bands.loc[:, "weight"] = np.where(rh_overlap[bands.index], (bands["bandEnd"] - gene_start) / span, bands["weight"])
            bands.loc[:, "weight"] = np.where(lh_overlap[bands.index], (gene_end - bands["bandStart"]) / span, bands["weight"])
            for j, band in bands.iterrows():
                spanning_genes.append((
                    band["chrom"],
                    band["band"],
                    band["bandStart"],
                    band["bandEnd"],
                    gene["geneName"],
                    gene_start,
                    gene_end,
                    band["weight"],
                ))
        del bands
    genome = pd.DataFrame(spanning_genes, columns=["chrom", "band", "bandStart", "bandEnd", "geneName", "txStart", "txEnd", "weight"])
    
    return genome.drop_duplicates(["chrom", "band", "geneName"])
