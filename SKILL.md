---
name: echidna
description: Use the sc-echidna Python package to train and inspect joint scRNA-seq and copy-number models, infer CNV and gene dosage effects, or work with its saved Pyro model. Apply to Echidna analysis and plotting tasks using AnnData and gene-indexed copy-number data.
---

# Echidna

Use `import echidna as ec`. The public analysis functions live under `ec.tl` (`echidna/tools`); plots live under `ec.pl` (`echidna/plot`, singular). Check those modules for signatures when adapting an example. The notebooks in `tutorials/` are the working examples: single timepoint, multiple timepoints, gene dosage inference, and direct model access.

## Prepare and train

- Supply an `AnnData` with raw counts in `adata.layers[config.counts_layer]` (usually `"counts"`), a timepoint column in `.obs` even for a single timepoint, and a cluster column in `.obs`. Set `EchidnaConfig(timepoint_label=..., counts_layer=..., clusters=...)` to match the data; the code default for `clusters` is `"pheno_louvain"`, while the tutorials use `"leiden"`.
- Supply bulk copy numbers as a gene-indexed `pandas.Series` for one timepoint or a gene-indexed `DataFrame` with one column per timepoint. Make gene names unique, remove missing values, and ensure useful overlap with `adata.var_names`. For multiple timepoints, make the copy-number columns correspond to the `.obs` timepoints. If their lexical order differs from biological order, call `ec.tl.set_sort_order(adata, [...])` before training.
- Train with `ec.tl.echidna_train(adata, W, config)`. It modifies `adata`: records matched genes and copy numbers in `.var`, adds the train/validation split and encoded clusters in `.obs`, and saves the configuration and run ID in `.uns["echidna"]`. Choose `n_steps`, `val_split`, `learning_rate`, `patience`, and model hyperparameters for the dataset. `patience=None` disables early stopping.

```python
import echidna as ec

config = ec.tl.EchidnaConfig(
    timepoint_label="timepoint",
    counts_layer="counts",
    clusters="leiden",
    n_steps=500,
)
ec.tl.echidna_train(adata, W, config)
ec.tl.echidna_clones(adata, threshold=0.1)
ec.pl.echidna(adata, color=["echidna_clones"])
ec.pl.ppc(adata, "X")
ec.pl.ppc(adata, "W")
```

## Inspect and extend a fitted run

- `ec.tl.echidna_clones` assigns `.obs["echidna_clones"]`; a positive `threshold` selects manual cutting. For multiple timepoints, call `ec.tl.echidna_status` after clone assignment to add `.obs["echidna_status"]`. Plot these annotations with `ec.pl.echidna` on an existing embedding.
- `ec.pl.ppc(adata, variable)` accepts one of `"X"`, `"W"`, `"c"`, `"eta"`, or `"cov"`. `ec.tl.sample(adata, variable)` accepts one of those strings or a list. `ec.tl.simulate(adata)` runs a simulation based on a fitted model. `ec.pl.plate_model(adata)` renders the model graph.
- For CNV inference, call `ec.tl.echi_cnv(adata, genome=...)`, then `ec.pl.plot_cnv(adata)` or `ec.tl.cnv_results(adata)`. Assign clones before `plot_cnv`. With no `genome`, `echi_cnv` downloads an hg38 annotation. A supplied genome needs at least `chrom` and `geneName` columns. For dosage effects, call `ec.tl.gene_dosage_effect(adata)` after CNV inference, then `ec.pl.plot_gene_dosage(adata, clusters=[...], timepoints=[...])`; these selectors are integer indices.
- For custom Pyro work, use `model = ec.tl.load_model(adata)`, `data = ec.tl.build_torch_tensors(adata, model.config)`, and `ec.tl.get_learned_params(model, data)`. The model exposes `eta_posterior`, `c_posterior`, `cov_posterior`, `model`, and `guide`. See `tutorials/4-echidna-model.ipynb` before changing model internals.

## Preserve fitted results

An `.h5ad` file alone does not contain the trained model. By default, `save_model` writes the model and Pyro parameter store beneath `./_echidna_models/<run_id>/`, relative to the process working directory. `load_model` uses the run ID in `adata.uns["echidna"]` to find those files. CNV and gene dosage results are also saved as external files, with paths in `adata.uns["echidna"]["save_data"]`. Keep these files with the AnnData when moving or reopening an analysis. Avoid `ec.tl.reset_echidna_memory()` unless deleting all saved runs is intended.
