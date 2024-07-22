# echidna.tools.hmm.py

import logging, os, re, sys

import numpy as np
import pandas as pd
import torch

from scipy.stats import ttest_ind
from hmmlearn import hmm

from echidna.tools.housekeeping import load_model
from echidna.utils import get_logger, ECHIDNA_GLOBALS

logger = get_logger(__name__)

def cnv_results(adata):
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

def infer_cnv(adata, genome: pd.DataFrame=None, **kwargs):
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
    echidna = load_model(adata)
    eta = echidna.eta_ground_truth
    # normalize out variance of each gene from eta
    eta = eta / torch.sqrt(torch.diag(torch.cov(eta.T)))
    eta = eta.T.detach().cpu().numpy()
    eta = pd.DataFrame(eta).set_index(
        adata[:, adata.var["echidna_matched_genes"]].var.index
    )
    eta.columns = [f"echidna_clone_{c}" for c in eta.columns]
    del echidna
    torch.cuda.empty_cache()
    
    if "chr" not in genome["band"].iloc[0]:
        genome["band"] = genome["chrom"] + "_" + genome["band"]
    genome = sort_chromosomes(genome)

    clone_cols = [f"echidna_clone_{c}" for c in range(eta.shape[1])]
    bands_eta_merge = genome.merge(
        eta, left_on="geneName",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    if "weight" in genome.columns:
        bands_eta_merge[clone_cols] = bands_eta_merge[clone_cols].mul(
            bands_eta_merge["weight"], axis="index"
        )
    bands_eta_means = bands_eta_merge.groupby("band")[clone_cols].mean()
    bands_eta_means = bands_eta_means.reindex(
        genome["band"].drop_duplicates()
    ).dropna()

    chrom_counts = sort_chromosomes(
        bands_eta_merge.groupby("chrom")["band"].nunique()
    ).cumsum()
    
    for c in clone_cols:
        bands_eta_means[f"states_{c}"] = _get_states(
            bands_eta_means.loc[:, c].values, **kwargs
        )
    
    infer_cnv_save_path = os.path.join(
        ECHIDNA_GLOBALS["save_folder"], "_echidna_cnv.csv"
    )
    bands_eta_means.to_csv(
        infer_cnv_save_path,
        index=True,
    )
    adata.uns["echidna"]["save_data"]["infer_cnv"] = infer_cnv_save_path
    logger.info(
        "Added `.uns['echidna']['save_data']['infer_cnv']` : Path to CNV inference results."
    )

def sort_chromosomes(df):
    def chrom_key(chrom):
        match = re.match(r"chr(\d+|X|Y)", chrom)
        if match:
            chrom = match.group(1)
            if chrom == "X":
                return (float("inf"), 1)  # X at the end but before Y
            elif chrom == "Y":
                return (float("inf"), 2)  # Y at the end
            else:
                return (int(chrom), 0)
        return (float("inf"), 3)  # Any unexpected value goes at the very end
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df["chrom_key"] = df["chrom"].apply(chrom_key)
        df_sorted = df.sort_values(by=["chrom_key", "bandStart", "txStart"]).drop(columns="chrom_key")
        return df_sorted
    elif isinstance(df, pd.Series):
        sorted_index = sorted(df.index, key=chrom_key)
        return df[sorted_index]    

def _get_states(vals, n_components=5, p_thresh=1e-5, neutral=2, transition=1, startprob=1, verbose=False):
    """Implements a gaussian hmm to call copy number states smoothing along the genome
    
    Parameters
    ----------
    vals: ordered copy number values (from bin_by_bands function)
    n_components: number of components to use for the hmm. We default to 5 for better sensitivity.
    p_thresh: cutoff to determine if a cluster should also be neutral
    neutral: the number to use for neutral. For Echidna, this is always 2. 
    transition: specifies the transmat_prior in the GaussianHMM function
    startprob: specifies the startprob_prior in the GaussianHMM function
    
    Returns
    -------
    """
    scores = []
    models = []

    idx = np.random.choice(range(100), size=10)
    for i in range(10): 
        model = hmm.GaussianHMM(
            n_components=n_components,
            random_state=idx[i],
            n_iter=100,
            transmat_prior=transition,
            startprob_prior=.8
        )
        model.fit(vals[:, None])
        models.append(model)
        scores.append(model.score(vals[:, None]))
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
    
    # Predict the most likely sequence of states given the model
    states = model.predict(vals[:, None])

    # Determine which state corresponds to which CN states
    tmp = pd.DataFrame({"vals": vals, "states": states})
    state_dict = tmp.groupby("states")["vals"].mean().reset_index()
    
    # Determine the state that is neutral (closest to 2)
    state_dict["sq_dist_to_neut"] = (state_dict["vals"] - neutral)**2
    state_dict["dist_to_neut"] = state_dict["vals"] - neutral
    state_dict = state_dict.sort_values(by="sq_dist_to_neut")

    # Check if any other states are neutral using a t-test
    neutral_state = [state_dict.iloc[0]["states"]]
    neutral_val = [state_dict.iloc[0]["vals"]]
    neutral_dist = tmp[tmp["states"] == neutral_state[0]]["vals"]

    for i, row in state_dict.iterrows():
        if row["states"] not in neutral_state:
            tmp_dist = tmp[tmp["states"] == row["states"]]["vals"]
            p_val = ttest_ind(neutral_dist, tmp_dist)[1]
            if p_val > p_thresh:
                neutral_state.append(row["states"])
                neutral_val.append(row["vals"])
    
    # Create a function to classify the states
    def classify_state(row):
        if row["states"] in neutral_state:
            return "neut"
        elif row["vals"] > max(neutral_val):
            return "amp"
        else:
            return "del"
    
    # Apply the classification function
    state_dict["CNV"] = state_dict.apply(classify_state, axis=1)
    
    # Map the CNV states back to the original states
    state_map = state_dict.set_index("states")["CNV"].to_dict()
    cnvs = list(map(state_map.get, states))

    return cnvs

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

