# MO-BICC Bench — *Multi-Omics Biomarker Cross-Cohort Benchmark*

*(pronounced “mo-bik”)*

**MO-BICC Bench** is a reproducible benchmark for evaluating **biomarker discovery (feature selection)** and **disease diagnosis (machine learning)** across **multi-omics** datasets and **independent cohorts**. It compares **simple and complex models**, plus **ensembles**, to answer a practical question: *which methods yield compact, robust, and transferable biomarker panels across studies?*

---

## Why MO-BICC?

* **Cross-cohort focus:** Train on Cohort A, test on Cohort B (and vice-versa) to measure **generalization**, not just in-sample performance.
* **Multi-omics integration:** mRNA, miRNA, DNA methylation (± proteomics) with **early / intermediate / late** integration options.
* **Biomarker realism:** Compact panels (e.g., *k* ∈ {5, 10, 20, 30, 50, 100, 200}), cost-aware scoring, and **stability** checks.
* **Transparent preprocessing:** Variance filtering, FDR-controlled statistics, and a **PC1 < 50%** rule to curb redundancy.
* **Reproducible by design:** Fixed seeds, YAML configs, environment lockfiles, and scriptable pipelines (Make/Snakemake).

---

## Scope

* **Tasks**

  * **Diagnosis (binary/multiclass)**: e.g., AD vs control, Braak strata, etc.
  * **Feature selection**: rankers, aggregators, modality-aware selection.
  * **Ensembling**: rank aggregation + model stacking/blending.
* **Datasets (examples)**

  * **ROSMAP**, **MSBB**, **Mayo RNAseq** for cross-cohort bulk omics.
  * Optional: **Banner/BLSA proteomics** (TMT), **AMP-AD** multi-omics, single-cell/sn-multiome for validation.
* **Modalities**

  * **mRNA**, **miRNA**, **DNA methylation** (± **proteomics**).
  * Probe/feature IDs harmonized where applicable (e.g., HM27 subset for methylation, gene symbol mapping).

---

## Benchmark Design

* **Splits**

  * **Within-cohort CV** for model/tuning.
  * **Cross-cohort transfer** (e.g., train ROSMAP → test MSBB; train Mayo → test ROSMAP).
* **Preprocessing (per MOGONET-style)**

  * Variance thresholds (defaults): **mRNA 0.1**, **methylation 0.001**, **miRNA = 0** (drop only zero variance).
  * **ANOVA + BH-FDR** on training folds to preselect informative features.
  * Choose *k* so **PC1 explains < 50%** of variance; scale each omics to **\[0,1]**.
* **Integration strategies**

  * **Early** (feature-level concatenation with scaling).
  * **Intermediate** (learn per-omic embeddings → fuse).
  * **Late** (per-omic models → meta-learner).
* **Panel sizes**

  * Evaluate **k = {5, 10, 20, 30, 50, 100, 200}** (customizable) with identical splits for fair comparison.

---

## Methods Covered

* **Simple baselines:** t-test / ANOVA + Logistic Regression; Ridge/LASSO; Naive Bayes.
* **Classical ML:** SVM (RBF/linear), Random Forest, XGBoost, LightGBM.
* **Deep / graph:** MLPs; MOGONET-style graph integration (optional).
* **Ensembles:**

  * **Rank aggregation** (e.g., RRA, Borda variants) for feature selection.
  * **Model stacking/blending** across modalities and panel sizes.

---

## Metrics

* **Predictive:** AUROC, AUPRC, Accuracy, F1, Balanced Accuracy, **calibration** (Brier, ECE).
* **Panel quality:** size (*k*), **cost proxies**, assay count, cross-platform feasibility.
* **Stability & overlap:** Jaccard / Kendall / CMC across CV folds & cohorts.
* **Biology:** pathway enrichment (GO/KEGG/Reactome), known-gene recovery.

---

## Quickstart

```bash
# Clone your repo (after you create it on GitHub):
git clone https://github.com/<you>/mo-bicc-bench.git
cd mo-bicc-bench

# (optional) Python env
conda env create -f environment.yml
conda activate omics-benchmark

# (optional) R deps
Rscript R/install_packages.R

# 1) Put your wide table at data/raw/input_wide.csv
#    Columns: "Sample ID", "Diagnosis", then features...

# 2) Prepare expr/meta for limma
python src/prepare_for_limma.py \
  --input data/raw/input_wide.csv \
  --expr data/processed/expr.csv \
  --meta data/processed/meta.csv \
  --group Diagnosis

# 3) Run limma
Rscript scripts/run_limma.R \
  data/processed/expr.csv data/processed/meta.csv \
  Diagnosis results/limma_results.csv Control
```

**Snakemake (optional)**
Edit `config/config.yaml`, then:

```bash
snakemake -j 4
```

---

## Configuration

```yaml
# config/config.yaml
sample_col: "Sample ID"
group_col: "Diagnosis"
reference_group: "Control"
# add cohort paths, modality flags, panel sizes, seeds, etc.
```

---

## Repository Layout

```
src/                # Python modules (prep, MOGONET-style preprocessing)
scripts/            # CLI scripts (R & Python; limma, synapse downloaders)
config/             # YAML configs for runs/pipelines
data/
  raw/              # input sources (ignored by git)
  processed/        # prepared matrices & splits
results/            # metrics, tables, selected panels
notebooks/          # EDA and figure prototypes
reports/            # manuscript-ready outputs
figs/               # plots saved from runs
R/                  # R scripts and installers
logs/               # run logs
```

---

## Reproducibility & Data Access

* **Do not commit controlled data.** Keep large files in `data/raw/` (git-ignored).
* Use Synapse/ADKP credentials to fetch cohorts; document exact versions and **checksums**.
* Fix **random seeds**, save **splits** and **selected feature lists** per run.

---

## Citation

If you use **MO-BICC Bench**, please cite:

> *MO-BICC Bench: A Multi-Omics Biomarker Cross-Cohort Benchmark for Feature Selection, Diagnosis, and Ensembling.*
> (Cyrille Mesue Njume et al., 2025) — *preprint coming soon.*

---

## License

MIT. See `LICENSE`.

---

## Contact

Questions, ideas, or contributions? Open an issue or reach out to **Cyrille Mesue Njume**.
