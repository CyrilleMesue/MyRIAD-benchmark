#!/usr/bin/env python
# coding: utf-8

# # Benchmarking of Biomarker Discovery Methods

# ### Import Packages

# In[11]:


import pandas as pd 
from modules.benchmarker import BiomarkerPipeline
import os 
from modules.utils import save_json
from pathlib import Path
import warnings 
warnings.filterwarnings('ignore') 

def get_view_column_splits(data: pd.DataFrame) -> dict:
    """
    Dynamically split columns into groups of 200.
    Adapts automatically for datasets with 201, 401, or 601 columns.
    """
    view_column_splits = {}
    step = 200  # size per chunk

    # Start from column index 1 to skip 'Sample ID' or first column
    total_cols = data.shape[1] - 1
    num_groups = (total_cols + step - 1) // step  # ceiling division

    for i in range(num_groups):
        start = 1 + i * step
        end = min(1 + (i + 1) * step, data.shape[1])
        view_column_splits[i + 1] = data.columns[start:end]

    return view_column_splits 

from collections import OrderedDict

def balance_view_column_splits(view_column_splits: dict[int, list]) -> dict[int, list]:
    """
    Ensures keys 1,2,3 exist. If any are missing, redistribute all unique elements
    across 1,2,3 as evenly as possible (no repetition, stable order).
    If all three are already present, return unchanged.
    If input has none of 1,2,3, returns {1:[], 2:[], 3:[]}.
    """
    target_keys = (1, 2, 3)
    present = [k for k in target_keys if k in view_column_splits]

    # If all present, no change
    if len(present) == 3:
        # Ensure only 1,2,3 are returned; shallow-copy lists
        return {k: list(view_column_splits.get(k, [])) for k in target_keys}

    # Gather unique elements in order of first appearance across present keys
    seen = set()
    pooled = []
    for k in present:
        for item in view_column_splits[k]:
            if item not in seen:
                seen.add(item)
                pooled.append(item)

    # If none present, return empty buckets
    if not pooled:
        return {1: [], 2: [], 3: []}

    # Compute near-equal sizes for 3 buckets
    n = len(pooled)
    base, rem = divmod(n, 3)
    sizes = [base + (1 if i < rem else 0) for i in range(3)]

    # Slice pooled into 3 buckets
    result_lists = []
    idx = 0
    for size in sizes:
        result_lists.append(pooled[idx: idx + size])
        idx += size

    return {1: result_lists[0], 2: result_lists[1], 3: result_lists[2]}


# ### Load Validation Datasets

# In[7]:


# Load validation datasets
ewas_meths = pd.read_csv('../data/val_data/EWAS_ATLAS_va_data.csv') 
ctd_pathways = pd.read_csv('../data/val_data/CTD_val_data_pathways.csv') 
ctd_genes = pd.read_csv('../data/val_data/CTD_val_data_genes.csv')
genecards_genes = pd.read_csv('../data/val_data/genecards_val_data.csv') 
hmdd_mirs = pd.read_csv('../data/val_data/HMDD_val_data_miRNA.csv')


# ### Pipeline Configurations

# In[26]:


# prepare experiments
experiments_paths = {
    "221": '../data/MayoRNASeq/prepared/gene_expression_and_proteomics_data.csv',
    "222": '../data/MayoRNASeq/prepared/metabolomics_and_gene_expression_data.csv',
    "223": '../data/MayoRNASeq/prepared/metabolomics_and_proteomics_data.csv'
}

for experiment_id, input_data_path in experiments_paths.items():
    p = Path(input_data_path).expanduser().resolve()
    dataset_name = p.parent.parent.name             # "ROSMAP", "BRCA", ...
    file_name = p.name                        # "miRNA_and_gene_expression_and_dna_methylation_data.csv"
    experiment_name = p.stem                       # without .csv

    # Correct if/elif/else
    if dataset_name == "ROSMAP":
        disease = "AD"
    elif dataset_name == "BRCA":
        disease = "BRCA"
    else:
        disease = "PSP"

    outdir = Path("../BENCHMARKING") / dataset_name / experiment_name
    outdir.mkdir(parents=True, exist_ok=True)

    more_and_mogonet_data_dir = Path("../more_and_mogonet_data") / dataset_name / experiment_name

    data = pd.read_csv(p, index_col=0)

    validation_datasets = {
        1: {"HMDD": hmdd_mirs[hmdd_mirs.Disease == disease]},
        2: {
            "CTD": ctd_genes[ctd_genes.Disease == disease],
            "CTD-pathways": ctd_pathways[ctd_pathways.Disease == disease],
            "GeneCards": genecards_genes[genecards_genes.Disease == disease],
        },
        3: {"EWAS-ATLAS": ewas_meths[ewas_meths.Disease == disease]},
    }

    model_list = [
        "Logistic Regression",
        "Random Forest",
        "XGBClassifier",
        "Decision Tree",
        "Gradient Boosting",
        "CatBoosting Classifier",
        "AdaBoost Classifier",
        "MLPClassifier",
        "SVC",
    ]

    rankers = [
        "lasso", "ridge", "elasticnet",
        "shap", "lime",
        "xgb_feature_importance", "randomforest_feature_importance",
        "rf_permutation_feature_importance", "xgb_permutation_feature_importance",
        "t_test", "boruta", "mannwhitneyu", "svm_rfe",
    ]
    rank_aggregators = ["mean", "min", "median", "rra", "geom.mean", "stuart", "mra"]
    weight_aggregators = ["mean", "max", "median", "geom.mean", "ta"]
    aggregators = [f"{m}_rank" for m in rank_aggregators] + [f"{m}_weight" for m in weight_aggregators]
    k_features = list(range(10, 101, 10))

    parameter_configs = {
        "test_size": 0.2,
        "n_folds": 5,
        "reference_class": "Control",
        "feature_type": "multiomics",
        "scoring": "f1",
        "method_cutoff_values": list(range(5, 101, 5)),
        "groundtruth_cutoff_values": list(range(50, 1001, 50)),
    }

    view_column_splits = balance_view_column_splits(get_view_column_splits(data))
    compute_more_mogonet = True

    pipe = BiomarkerPipeline(
        data=data,
        rankers=rankers,
        aggregators=aggregators,
        models=model_list,
        k_features=k_features,
        labels_column_name="Diagnosis",
        parameter_configs=parameter_configs,
        view_column_splits=view_column_splits,
        more_and_mogonet_data_dir=str(more_and_mogonet_data_dir),
        validation_datasets=validation_datasets,
        pathway_validation_sets=["CTD-pathways"],
        experiment_id=experiment_id,
    )

    pipe.run_all(compute_more_mogonet=compute_more_mogonet)

    save_json(outdir / "ML-Baseline.json", pipe.results["ML:Baseline"])
    pd.DataFrame(pipe.results["ranked_lists"]).to_csv(outdir / "BiomarkerRanks.csv", index=False)
    pd.DataFrame(pipe.results["biomarker-validation"]).to_csv(outdir / "Biomarker-validation-results.csv", index=False)
    pipe.results["cross_val_results"].to_csv(outdir / "cross-validation-results.csv", index=False)