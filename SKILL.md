---
name: echidna
description: Analyze matched single-cell RNA-seq and bulk WGS copy-number data with sc-echidna. Use for clonal reconstruction, longitudinal clone dynamics, CNV and gene dosage analysis, posterior checks, plotting, and saved Pyro models.
---

# Echidna

Install the package as `echidna` with Bioconda or `sc-echidna` with pip, then use `import echidna as ec`. Analysis functions are under `ec.tl`; plotting functions are under `ec.pl`. Echidna jointly fits single-cell RNA counts and gene-level bulk WGS copy numbers from one or several matched timepoints. Its Bayesian model infers per-cluster copy-number profiles (`eta`), latent expression-related values (`c`), and covariance among clusters, which link chromosomal alterations to tumor cell states. The [Echidna paper](https://www.biorxiv.org/content/10.1101/2024.12.15.628568v1) motivates this through phenotypic plasticity and tumor evolution.

## Choose a workflow

| Goal | Use |
| --- | --- |
| Reconstruct clones from expression and copy-number profiles | Fit with `ec.tl.echidna_train`, assign clone labels with `ec.tl.echidna_clones`, inspect the tree with `ec.pl.dendrogram`, and display clones with `ec.pl.echidna`. |
| Follow clones through serial samples | Set biological timepoint order with `ec.tl.set_sort_order`, fit the multi-timepoint model, then use `ec.tl.echidna_status` to label growing, stable, and shrinking clones. |
| Locate copy-number alterations | Use `ec.tl.echi_cnv` with a gene-to-chromosome annotation, `ec.tl.cnv_results` for a gene-indexed table, and `ec.pl.plot_cnv` for a chromosome view. `ec.pl.plot_eta` shows the fitted `eta` heatmap. |
| Quantify gene dosage effects | Run `ec.tl.gene_dosage_effect` after CNV inference, then use `ec.pl.plot_gene_dosage` to compare effect scores with fitted copy-number shifts by cluster and timepoint. |
| Inspect fit and extend the model | Use `ec.pl.ppc` for posterior predictive plots, `ec.tl.sample` for posterior draws, `ec.tl.simulate` for a fitted simulation, or `ec.tl.load_model` for direct Pyro work. |

## Prepare matched inputs

Start with a processed `scanpy.AnnData`:

- `.layers["counts"]`: raw, nonnegative RNA counts.
- `.obs[<clusters>]`: cluster labels. Echidna fits one `eta` profile per cluster, so these are the units that get grouped into clones.
- `.obs["timepoint"]`: one label per cell. Set it to a constant for a single timepoint.
- `.obsm["X_umap"]`: used by `ec.pl.echidna` by default.

Change `EchidnaConfig(timepoint_label=..., counts_layer=..., clusters=...)` when the dataset uses other names. The `clusters` default is `"pheno_louvain"`, so always set it explicitly. `ec.tl.pre_process(adata)` stores raw counts in `.layers["counts"]`, runs PhenoGraph Leiden clustering (labels land in `.obs["pheno_leiden"]`, so pass `clusters="pheno_leiden"`), and computes a UMAP. Add `.obs["timepoint"]` yourself; preprocessing does not create it.

Supply bulk copy numbers as a `pandas.Series` indexed by gene for one timepoint, or a `pandas.DataFrame` indexed by gene with one column per timepoint. Gene names must match `adata.var_names` and be unique (`echidna_train` raises on duplicate indices). Keep only genes shared by both inputs.

For multiple timepoints, W columns are not matched to `.obs["timepoint"]` by exact name. They are sorted by the first label in the `set_sort_order` list that appears as a **substring** of the column name. For example, `"R310_on2_count"` is placed with `"on"`. Pick labels so each column contains exactly the intended one. `"pre"` is a substring of `"post1_pre2"`, and `"on"` matches both `"on2"` and `"post1_on2"`. The number of W columns must equal the number of distinct timepoints in `.obs`.

## Single timepoint example

This uses the repository's `demo_data/` files. They are stored in Git LFS, so run `git lfs pull` in a fresh clone, and they are not included in the pip or Bioconda package. Gunzip the `.h5ad.gz` first (`gunzip -k demo_data/R310_scDNA_ST.h5ad.gz`); pandas reads the `.csv.gz` directly. For other data, replace the paths and column names.

```python
import pandas as pd
import scanpy as sc
import echidna as ec

adata_st = sc.read_h5ad("demo_data/R310_scDNA_ST.h5ad")
adata_st.obs["timepoint"] = "single_tp"
# Columns: unnamed index, "0" = gene name, "1" = copy number
w_st = pd.read_csv("demo_data/R310_W_ST.csv.gz", index_col=0).set_index("0")["1"]
w_st = w_st.loc[~w_st.index.duplicated(keep=False)].dropna()
shared_genes = adata_st.var_names.intersection(w_st.index)
adata_st = adata_st[:, shared_genes].copy()
w_st = w_st.reindex(adata_st.var_names)

config_st = ec.tl.EchidnaConfig(
    timepoint_label="timepoint",
    counts_layer="counts",
    clusters="leiden",
    inverse_gamma=True,
    eta_mean_init=w_st.mean(),
    q_corr_init=0.1,
    q_shape_rate_scaler=10.0,
    lkj_concentration=1.0,
    patience=500,
    n_steps=500,
)
ec.tl.echidna_train(adata_st, w_st.copy(), config_st)
ec.tl.echidna_clones(adata_st, method="elbow")
ec.pl.dendrogram(adata_st)
ec.pl.echidna(adata_st, color=["echidna_clones"])
ec.pl.ppc(adata_st, "X")
ec.pl.ppc(adata_st, "W")
```

## Multiple timepoints example

`demo_data/R310_scRNA_MT.h5.gz` has `pre` and `on` in `.obs["timepoint"]`. `demo_data/R310_W_MT.csv.gz` has `geneName`, `R310_pre_count`, and `R310_on2_count` columns; the `on2` column is matched to `on` by substring. Gunzip the `.h5.gz` first.

```python
import pandas as pd
import scanpy as sc
import echidna as ec

adata_mt = sc.read_h5ad("demo_data/R310_scRNA_MT.h5")
w_mt = pd.read_csv("demo_data/R310_W_MT.csv.gz").set_index("geneName")
w_mt = w_mt.loc[~w_mt.index.duplicated(keep=False)].dropna()
shared_genes = adata_mt.var_names.intersection(w_mt.index)
adata_mt = adata_mt[:, shared_genes].copy()
w_mt = w_mt.reindex(adata_mt.var_names)

ec.tl.set_sort_order(adata_mt, ["pre", "on"])

config_mt = ec.tl.EchidnaConfig(
    timepoint_label="timepoint",
    counts_layer="counts",
    clusters="leiden",
    inverse_gamma=False,
    patience=None,
    n_steps=500,
    val_split=0.1,
    learning_rate=0.1,
    q_corr_init=1e-2,
    q_shape_rate_scaler=10.0,
    eta_mean_init=2.0,
    lkj_concentration=1.0,
)
ec.tl.echidna_train(adata_mt, w_mt.copy(), config_mt)
ec.tl.echidna_clones(adata_mt, threshold=0.1)
ec.tl.echidna_status(adata_mt, threshold=0.6)
ec.pl.echidna(adata_mt, color=["echidna_clones", "echidna_status"])
```

## Choosing settings

The example settings came from tuning on the demo data. Treat them as starting points, not rules.

- **`inverse_gamma`**: use `True` for small or noisy single-timepoint data, where it tends to be more stable. Use `False` (the default) when there are more cells and timepoints.
- **`eta_mean_init`**: start near the dataset's typical copy number. Use `w.mean()` when the scale is unusual; otherwise use `2.0` (diploid).
- **`q_corr_init`**: the initial scale of the variational correlation. `0.01` (default) starts tighter than `0.1`.
- **`n_steps` / `patience`**: the default is `n_steps=10000`; `500` is a quick demo budget. With `verbose=True`, training shows a progress bar and then plots training and validation loss on a log scale. Treat the fit as converged once both curves flatten. If they haven't, raise `n_steps`. `patience=N` stops after N steps without validation improvement; `None` or `0` disables early stopping.
- **Clone threshold**: `echidna_clones` builds a dendrogram over the clusters' `eta` profiles (default metric `"smoothed_corr"`). For new data, start with `method="elbow"` or `method="cophenetic"`. Check the tree with `ec.pl.dendrogram(adata)` (`elbow=True` shows the elbow curve), then set a manual `threshold=` cut if needed. Any `threshold > 0` switches to manual mode.
- **Status threshold**: `echidna_status(threshold=...)` computes `log2(fraction at current timepoint / fraction at previous timepoint)` for each clone. Scores above `threshold` are labelled growing, below `-threshold` shrinking, and anything else stable. Fractions use all cells in `adata`, including those marked `discard` during fitting.

## What training changes on the AnnData

- `.obs[<clusters>]` is **replaced with integer codes**. If the labels weren't already integers, the originals are copied to `.obs[<clusters> + "_categorical"]`. Cluster indices in later calls, such as `plot_gene_dosage(clusters=[...])`, refer to these codes.
- `.obs["echidna_split"]` holds `train`, `validation`, or (multi-timepoint only) `discard`. Multi-timepoint training **downsamples every timepoint to the size of the smallest one**. The extra cells are marked `discard` and not used to fit.
- `.var["echidna_matched_genes"]` and `.var["echidna_W_*"]` hold the matched genes and copy numbers.
- `.uns["echidna"]` holds the config, `run_id`, `timepoint_order`, and `save_data` paths.
- The W passed in is modified in place: NaNs are dropped and columns are renamed to `echidna_W_*`. Pass `w.copy()` to keep the original.

## CNV and gene dosage analysis

After fitting the multi-timepoint example:

```python
ec.tl.echi_cnv(adata_mt)
ec.pl.plot_cnv(adata_mt)
cnv_by_gene = ec.tl.cnv_results(adata_mt)
ec.tl.gene_dosage_effect(adata_mt)
ec.pl.plot_gene_dosage(
    adata_mt, clusters=[0, 3, 4], timepoints=[0, 1], quantile=0.8
)
```

`echi_cnv` fits a Gaussian mixture per cluster to find its neutral (baseline) copy-number level, then runs an HMM over genes in genomic order to call states. Without `genome`, it downloads the hg38 GENCODE V46 annotation from UCSC, which **needs internet access**. For other genomes or offline use, pass a `pandas.DataFrame` with:

- `chrom` in `chr1`…`chr22`, `chrX`, `chrY` form
- `geneName`
- `txStart`

Genes are ordered within each chromosome by `txStart`. Supply it for reliable genomic ordering; without it, order within a chromosome is not guaranteed.

Parameters that are commonly adjusted (defaults shown):

- `gaussian_smoothing=True`, `smoother_sigma=6`, `smoother_radius=8`: smooth `eta` along the genome before HMM calling.
- `filter_genes=True`, `filter_quantile=0.7`: restrict to higher-variance genes. The GMM neutral estimate always uses this filter; `filter_genes=False` keeps all matched genes for the HMM.
- `n_gmm_components=5`, `n_hmm_components=5`: mixture components and candidate HMM states.
- Passed through `**kwargs`: `n_iter=100`, `transmat_prior=1`, and `startprob_prior=1` go to the HMM; `plot_gmm=True` plots each cluster's mixture fit to check the chosen neutral component.

`gene_dosage_effect` uses posterior draws and the neutral baseline to save a gene × timepoint × cluster effect tensor. `plot_gene_dosage` plots absolute effect score against the fitted `eta` shift from baseline for genes above the `quantile` expression-variance cutoff. It runs `gene_dosage_effect` itself if no cached result exists. `clusters` and `timepoints` are integer indices: cluster codes as above, and timepoints in `set_sort_order` order. Choose indices present in the fitted data.

## Model checks and direct access

- `ec.pl.ppc(adata, "X")` and `ec.pl.ppc(adata, "W")` compare fitted predictions with the observed RNA counts and bulk copy numbers.
- `"c"`, `"eta"`, and `"cov"` show the latent quantities; `ec.pl.ppc(adata, "cov", corr=True)` shows correlations.
- `ec.tl.sample(adata, ["X", "W"])` returns a list of samples; a single name such as `"eta"` returns that variable's samples.
- `ec.tl.simulate(adata)` refits on sampled `X` and `W` to check whether the fitted parameters can be recovered.
- `ec.pl.plate_model(adata)` renders the Pyro model graph.

For custom Pyro analysis:

```python
model = ec.tl.load_model(adata_mt)
data = ec.tl.build_torch_tensors(adata_mt, model.config)
eta, c, cov = model.eta_posterior, model.c_posterior, model.cov_posterior
learned_params = ec.tl.get_learned_params(model, data)
model.model(*data)
model.guide(*data)
```

`model.model` and `model.guide` follow Pyro conventions. They are wrapped in `poutine.scale` and switch between the single- and multi-timepoint versions based on the config.

## Saving and gotchas

- Models are saved under `./_echidna_models/<run_id>/` (`echidna_model.pt` and `echidna_model_param_store.pt`), with `<run_id>` in `adata.uns["echidna"]["run_id"]`. The path is **relative to the Python working directory**. Starting a notebook from a different folder makes `load_model` fail unless you pass `save_folder=`. CNV tables and dosage tensors are saved there too, and their paths are recorded in `adata.uns["echidna"]["save_data"]`. Keep the `.h5ad` and the run directory together.
- Retraining on the same AnnData overwrites its current run. `ec.tl.save_model(adata, model)` saves a model you changed by hand.
- `ec.tl.reset_echidna_memory()` **deletes the entire `./_echidna_models/` folder**, including every saved run. Don't call it unless the user asks.
- `device` defaults to `"cuda"` when available. Training and `load_model` call `torch.set_default_device` and `torch.set_default_dtype(float32)` globally, which affects other torch code in the same session.
- Each `load_model` call clears Pyro's global param store.
- `ec.tl.filter_low_var_genes(adata, quantile=0.75)` optionally removes low-variance genes before training.
