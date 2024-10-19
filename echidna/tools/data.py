# echidna.tools.data.py

import pyro
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import warnings, requests, json, re

from echidna.tools.utils import EchidnaConfig, _custom_sort
from echidna.utils import get_logger

logger = get_logger(__name__)

def pre_process(
    adata: sc.AnnData, 
    num_genes: int=None, 
    target_sum: float=None, 
    exclude_highly_expressed: bool=False, 
    n_comps: int=15, 
    phenograph_k: int=60, 
    n_neighbors: int=15,
    filepath: str=None,
    ) -> sc.AnnData:
    """Basic pre-processing pipeline. Choose parameters according to your data.
    
    Parameters
    ----------
    adata : sc.AnnData
        Unprocessed annotated data matrix.
    num_genes : int
        Number of highly expressed genes to use. Pass None if using
        all (recommended).
    target_sum : 
        Normalize to this total, defaults median library size.
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes.
    n_comps : int
        Number of principal components to use for PCA.
    phenograph_k : int
        Number of nearest neighbors to use in first step of graph
        construction.
    n_neighbors : int
        Number of nearest neighbors for UMAP.
    filepath : str
        If defined, will save the processed AnnData to the specified location.
        
    Returns
    -------
    adata : sc.AnnData
        Processed annotated data matrix.
    """
    from scipy.sparse import csr_matrix

    adata.X = adata.X.astype("float32")

    # highly variable genes
    if num_genes is not None:
        x_log = sc.pp.log1p(adata, copy=True, base=10)
        sc.pp.highly_variable_genes(x_log, n_top_genes=num_genes)
        adata = adata[:, x_log.var["highly_variable"]]

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.calculate_qc_metrics(adata, inplace=True, layer="counts")

    # store the current "total_counts" under original_total_counts, 
    # which will not automatically be updated by scanpy in subsequent filtering steps
    adata.obs["original_total_counts"] = adata.obs["total_counts"].copy()

    # log10 original library size
    adata.obs["log10_original_total_counts"] = np.log10(adata.obs["original_total_counts"]).copy()

    # Normalize by median library size
    if target_sum is None:
        target_sum = np.median(adata.obs["original_total_counts"])
    sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

    # log transform + 1 and updates adata.X
    sc.pp.log1p(adata)

    logger.info("Performing PCA...")
    sc.tl.pca(adata, n_comps=n_comps)
    logger.info("Calculating phenograph clusters...")
    sc.external.tl.phenograph(adata, clustering_algo="leiden", k=phenograph_k, seed=1)

    logger.info("Performing nearest neighbors search and calculating UMAP...")
    sc.pp.neighbors(adata, random_state=1, n_neighbors=n_neighbors)
    sc.tl.umap(adata, random_state=1)

    for sparse_mtx in adata.obsp:
        adata.obsp[sparse_mtx] = csr_matrix(adata.obsp[sparse_mtx])
    if filepath is not None:
        adata.write_h5ad(filepath)
    return adata

def filter_low_var_genes(
    adata: sc.AnnData,
    quantile: float=0.75,
    var_threshold: float=None,
    indices: bool=False,
) -> sc.AnnData:
    gene_variances = adata.X.var(axis=0)
    
    if var_threshold is not None:
        gene_filter = gene_variances > var_threshold # / (adata.X.mean(axis=0) + 1e-8)
    elif var_threshold is None:
        var_threshold = np.quantile(gene_variances, quantile)
        gene_filter = gene_variances > var_threshold
    if indices:
        return gene_filter
    return adata[:, gene_filter]

def train_val_split(adata, config):
    rng = np.random.default_rng(config.seed)
    
    tmp_idx, i = "index", 0
    while tmp_idx in adata.obs.columns:
        tmp_idx = f"index{i}"
        i+=1
    adata.obs.reset_index(names=tmp_idx, inplace=True)
    
    if config._is_multi:
        adata_vc = adata.obs[config.timepoint_label].value_counts()

        n_val = int(config.val_split * adata_vc.min())
        smallest_tp = adata_vc.index[adata_vc.argmin()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            adata.obs["echidna_split"] = "train"
        for tp in adata_vc.index:
            tp_filter = adata.obs[config.timepoint_label] == tp
            cur_tp_index = adata.obs[tp_filter].index

            val_idx = rng.choice(cur_tp_index, n_val, replace=False)

            n_discard = len(cur_tp_index) - adata_vc.min()
            if n_discard > 0:
                train_idx = np.setdiff1d(cur_tp_index, val_idx)
                discard_idx = rng.choice(train_idx, n_discard, replace=False)
                adata.obs.loc[discard_idx, "echidna_split"] = "discard"

            adata.obs.loc[val_idx, "echidna_split"] = "validation"

        adata.obs["echidna_split"] = adata.obs["echidna_split"].astype("category")
    else:
        n_val = int(config.val_split * adata.shape[0])
        val_idx = rng.choice(adata.obs.index, n_val, replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            adata.obs["echidna_split"] = "train"
        adata.obs.loc[adata.obs.index[val_idx], "echidna_split"] = "validation"
        adata.obs["echidna_split"] = adata.obs["echidna_split"].astype("category")
        
    adata.obs.set_index(tmp_idx, inplace=True)
    
    logger.info(
        "Added `.obs['echidna_split']`: the Echidna train/validation split.\n"
        f" {n_val} cells in validation set."
    )

    return adata

def create_z_pi(adata, config):
    config = config.to_dict() if isinstance(config, EchidnaConfig) else config
    if config["clusters"] not in adata.obs.columns:
        raise ValueError(f"{config['clusters']} clustering not in AnnData obs")
    if not bool(config["_is_multi"]):
        z_obs_series = adata.obs[config["clusters"]].values
        pi_obs_series = np.unique(z_obs_series, return_counts=True)[1] / len(z_obs_series)
        z_obs = torch.from_numpy(np.array(z_obs_series)).to(torch.float32).to(config["device"])
        pi_obs = torch.from_numpy(pi_obs_series).to(torch.float32).to(config["device"])
    else:
        adata_tmp = adata[adata.obs["echidna_split"]!="discard"].copy()
        timepoints = np.unique(adata_tmp.obs[config["timepoint_label"]])
        if "timepoint_order" in adata_tmp.uns["echidna"]:
            timepoints = _custom_sort(timepoints, adata.uns["echidna"]["timepoint_order"])
        z = []
        pi = []
        for t in timepoints:
            z_tmp = adata_tmp.obs[adata_tmp.obs[config["timepoint_label"]]==t][config["clusters"]].values
            pi_tmp = torch.zeros(int(config["num_clusters"]), dtype=torch.float32)
            indices, counts = np.unique(z_tmp, return_counts=True)
            pi_proportions = counts / len(z_tmp)
            for i, p in zip(indices, pi_proportions): pi_tmp[i] = p
            z.append(torch.tensor(z_tmp, dtype=torch.int64))
            pi.append(pi_tmp)
        z_obs = torch.stack(z).to(torch.float32).to(config["device"])
        pi_obs = torch.stack(pi).to(torch.float32).to(config["device"])
    return pi_obs, z_obs

def match_genes(adata, Wdf):
    """Matches genes between AnnData and W.

    Parameters
    ----------
        adata : sc.AnnData
            Annotated data matrix.
        Wdf : pd.DataFrame
            DataFrame containing copy number counts, indexed by genes.
    """
    if Wdf.index.duplicated().any():
        raise ValueError("Duplicate indices in W. Make sure W is uniquely indexed by genes.")
    
    Wdf.dropna(inplace=True)
    matched_genes = adata.var.index.intersection(Wdf.index)
    adata.var["echidna_matched_genes"] = np.where(adata.var.index.isin(matched_genes), True, False)
    logger.info("Added `.var[echidna_matched_genes]` : Labled True for genes contained in W.")
    if len(Wdf.shape) > 1:
        Wdf.columns = [f"echidna_W_{c}" if "echidna_W_" not in c else c for c in Wdf.columns]
        col_name = Wdf.columns
    elif len(Wdf.shape) == 1:
        if Wdf.name is None:
            Wdf.name = "echidna_W_count"
        else:
            Wdf.name = "echidna_W_" + Wdf.name if "echidna_W_" not in Wdf.name else Wdf.name
        col_name = [Wdf.name]
    if len(np.intersect1d(adata.var.columns, col_name)) == 0:
        adata.var = adata.var.merge(Wdf, left_index=True, right_index=True, how="left")
        # display(merged_var)
        # adata = adata[:, ]
        for c in col_name:
            logger.info(f"Added `.var[{c}]` : CN entries for genes contained in W.")

def build_torch_tensors(adata, config):
    """
    Takes anndata and builds Torch tensors.
    """
    Wdf = adata.var[[c for c in adata.var.columns if "echidna_W_" in c]].dropna()
    if config._is_multi and "timepoint_order" in adata.uns["echidna"]:
        Wdf = _custom_sort(Wdf, adata.uns["echidna"]["timepoint_order"])
    W_obs = torch.from_numpy(Wdf.values).to(torch.float32).to(config.device)
    
    if W_obs.shape[-1] != config.num_timepoints:
        raise ValueError(
            "Number of W columns found in AnnData does not match"
            " number of timepoints, drop excess if needed."
            " Check columns in `.var` :", list(Wdf.columns)
        )
    if config._is_multi:
        W_obs = W_obs.T
        adata = adata[adata.obs["echidna_split"]!="discard"]
        tps = adata.obs[config.timepoint_label].unique()
        if "timepoint_order" in adata.uns["echidna"]:
            tps = _custom_sort(tps, adata.uns["echidna"]["timepoint_order"])
        x_list = [adata[tp == adata.obs[config.timepoint_label]].layers[config.counts_layer] for tp in tps]
        X_obs = torch.from_numpy(np.array(x_list)).to(torch.float32).to(config.device)
    elif not config._is_multi:
        W_obs = W_obs.flatten()
        X_obs = torch.from_numpy(adata.layers[config.counts_layer]).to(torch.float32).to(config.device)
    if config.clusters:
        pi_obs, z_obs = create_z_pi(adata, config)
        return X_obs, W_obs, pi_obs, z_obs
    return X_obs, W_obs

def fetch_genome():
    genes = pd.read_csv(
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/wgEncodeGencodeCompV46.txt.gz",
        delimiter="\t",
        header=None,
        names=["gene_id", "transcript_id", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds", "score", "geneName", "cdsStartStat", "cdsEndStat", "exonFrames"],
    )
    genes = genes[["chrom", "geneName", "txStart", "txEnd"]].drop_duplicates()
    genes_gb = genes.groupby(["chrom", "geneName"])
    min_tx = genes_gb.min()["txStart"].reset_index()
    max_tx = genes_gb.max()["txEnd"].reset_index()
    gene_coords = min_tx.merge(max_tx, on=["chrom", "geneName"])
    
    ensembl_to_gene_name = ensembl_conversion(list(gene_coords[gene_coords.geneName.str.startswith("ENS")].geneName))
    if ensembl_to_gene_name:
        def map_gene_names(row, mapping):
            new_name = mapping.get(row['geneName'], None)
            return new_name if new_name is not None else row["geneName"]

        gene_coords["geneName"] = gene_coords.apply(lambda row: map_gene_names(row, ensembl_to_gene_name), axis=1)
    
    return sort_chromosomes(gene_coords)

def ensembl_conversion(ensembl_genes):
    response = requests.post(
        "https://biotools.fr/mouse/ensembl_symbol_converter/",
        data={
            'api': 1,
            'ids': json.dumps(ensembl_genes)
        },
    )
    if response.status_code == 200:
        output = response.json()
    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        return
    return output

def range_subset_pct(
    gene_range, chrom_region
):
    start, end = gene_range
    chrom_start, chrom_end = chrom_region
    overlap = max(0, min(end, chrom_end) - max(start, chrom_start))
    return overlap / (end - start) if (end - start) > 0 else 0

def range_subset(
    gene_range, chrom_region
):
    start, end = gene_range
    chrom_start, chrom_end = chrom_region
    return 1 if start >= chrom_start and end <= chrom_end else 0

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
        sort_cols = np.intersect1d(["bandStart", "txStart"], df.columns)
        df_sorted = df.sort_values(by=["chrom_key"] + list(sort_cols)).drop(columns="chrom_key")
        return df_sorted
    elif isinstance(df, pd.Series):
        sorted_index = sorted(df.index, key=chrom_key)
        return df[sorted_index]

def get_w(
    ichor: pd.DataFrame,
    genome: pd.DataFrame,
    timepoint: str,
    verbose: bool=False,
    chrom_label: str="chrom",
    gene_start: str="txStart",
    gene_end: str="txEnd",
    gene_name: str="geneName",
    weighted: bool=True,
):
    w = []
    genome = genome.drop_duplicates()
    ichor["chr"] = ichor["chr"].apply(lambda x: x if x.startswith("chr") else "chr" + x)
    
    func = range_subset_pct if weighted else range_subset
    
    seen = defaultdict(float)
    for _, cn_row in ichor.iterrows():

        chrom = genome[genome[chrom_label].astype(str) == cn_row["chr"]]
        chrom_region = (cn_row.start, cn_row.end)
        
        if verbose: print(f"------- Matching chromosome {cn_row.chr} and range {chrom_region}...")
        for _, gene in chrom.iterrows():
            gene_range = (gene[gene_start], gene[gene_end])
            pct = func(gene_range, chrom_region)
            
            if gene[gene_name] not in seen:
                seen[gene[gene_name]] = pct * cn_row["copy.number"]
            else:
                seen[gene[gene_name]] += pct * cn_row["copy.number"]

    W = pd.DataFrame({"geneName": list(seen.keys()), f"{timepoint}_count": list(seen.values())})
    return W