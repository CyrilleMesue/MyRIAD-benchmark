# Import Packages 
import pandas as pd
import random
from itertools import combinations
import re
import os
import shutil
import numpy as np
from modules.functionalEnrichment.gProfiler import gProfilerFunctionerEnricher 
from modules.more_mogonet_utils import generate_and_run_training_script, generate_and_run_feat_imp_script 



def top_up_selected_features(selected_features, feat_list, features, min_count=2):
    """
    Ensures at least `min_count` features from `feat_list` are in `selected_features`.
    If not, tops it up using ranked `features` (e.g., global feature ranking).

    Args:
        selected_features (list): Already selected features.
        feat_list (list): Reference list (e.g., ground truth or pathway genes).
        features (list): Ranked list of global features (e.g., by importance).
        min_count (int): Minimum required overlap.

    Returns:
        list: Updated selected_features with at least `min_count` intersecting feat_list
    """
    selected_set = set(selected_features)
    feat_set = set(feat_list)
    intersection = selected_set & feat_set

    if len(intersection) >= min_count:
        return selected_features  # Already sufficient

    # Add more features from `features` that are in `feat_list` and not already selected
    to_add = []
    for f in features:
        if f in feat_set and f not in selected_set:
            to_add.append(f)
            if len(intersection) + len(to_add) >= min_count:
                break

    return selected_features + to_add
    

def prepare_data_from_single_dataframe(
    df: pd.DataFrame,
    label_col: str,
    view_column_splits: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: str
):
    """
    Prepares and saves omics data and labels from a single dataframe into the format 
    required by MORE and MOGONET.

    Args:
        df (pd.DataFrame): Full dataframe containing labels and all features.
        label_col (str): Column name for labels.
        view_column_splits (dict): Dictionary mapping view number to list of column names.
                                   e.g., {1: ['gene1', 'gene2'], 2: [...], 3: [...]}
        train_idx (np.ndarray): Row indices for training samples.
        test_idx (np.ndarray): Row indices for testing samples.
        output_dir (str): Target folder to write files.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract labels
    labels_tr = df[label_col].iloc[train_idx].astype(int)
    labels_te = df[label_col].iloc[test_idx].astype(int)

    np.savetxt(os.path.join(output_dir, "labels_tr.csv"), labels_tr.values, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(output_dir, "labels_te.csv"), labels_te.values, fmt="%d", delimiter=",")

    # For each view, save data and feature names
    for view_num, columns in view_column_splits.items():
        df_view = df[columns]

        # Train/test split
        X_tr = df_view.iloc[train_idx]
        X_te = df_view.iloc[test_idx]

        np.savetxt(os.path.join(output_dir, f"{view_num}_tr.csv"), X_tr.values, delimiter=",")
        np.savetxt(os.path.join(output_dir, f"{view_num}_te.csv"), X_te.values, delimiter=",")

        # Save feature names
        pd.Series(columns).to_csv(os.path.join(output_dir, f"{view_num}_featname.csv"), index=False, header=False)


def compute_pathways(gene_list):

    parameters = {
        'organism': 'hsapiens',
        'query': gene_list, 
        'sources': ['KEGG', 'REAC'],
        'ordered': True,
        'user_threshold': 0.05,
        'no_evidences': False
    }
    
    try:
        enricher = gProfilerFunctionerEnricher(parameters)
        enricher.remove_parent_terms()
        result_df = enricher.results_table()  
        if result_df.shape[0] != 0: 
            return result_df[["name", "p_value"]][result_df.p_value <= 0.05].name.to_list()
        else:
            return []
    except:
        return []
        

def generate_random_combinations(d, sizes=(5, 7, 9), sample_size=100):
    keys = list(d.keys()) 
    
    all_combos = {}

    # Generate all possible subsets first
    for size in sizes:
        if size <= len(keys):
            for combo in combinations(keys, size):
                combo_name = "|".join(combo)
                all_combos[combo_name] = {k: d[k] for k in combo}

    # Randomly select 100 subsets 
    selected_keys = random.sample(list(all_combos.keys()), sample_size)
    selected_combos = {k: all_combos[k] for k in selected_keys}

    return selected_combos 


def extract_metrics(stdout_text: str) -> dict:
    """
    Extract classification metrics from stdout text.
    """
    pattern = (
        r"accuracy: ([\d.]+) \| f1: ([\d.]+) \| roc_auc: ([\d.]+) \| precision: ([\d.]+) "
        r"\| recall: ([\d.]+) \| specificity: ([\d.]+) \| npv: ([\d.]+) \| lr_plus: ([\d.]+) \| lr_minus: ([\d.]+)"
    )
    
    match = re.search(pattern, stdout_text)
    if not match:
        return {}
    
    return {
        "accuracy": float(match.group(1)),
        "f1": float(match.group(2)),
        "roc_auc": float(match.group(3)),
        "precision": float(match.group(4)),
        "recall": float(match.group(5)),
        "specificity": float(match.group(6)),
        "npv": float(match.group(7)),
        "lr_plus": float(match.group(8)),
        "lr_minus": float(match.group(9)),
    }


def run_more_and_mogonet_cv_pipeline(
    cross_validation_indices,
    labels,
    data_views,
    view_list,
    data_folder,
    script_path="./Code/temp_train_test.py",
    repo_path="../MORE", 
    model_type = "more", 
    dim_he_list:list = None
):
    """
    Runs MORE or MOGONET training and evaluation across CV folds, including safe directory change.

    Args:
        cross_validation_indices (list): List of (train_idx, test_idx) tuples.
        labels (np.ndarray): Label vector.
        data_views (list of pd.DataFrame): Omics data per view.
        view_list (list): List of view numbers [1, 2, 3].
        data_folder (str): Output folder to save split files.
        script_path (str): Path to training script.
        more_repo_path (str): Path to the MORE GitHub repo (for os.chdir).

    Returns:
        dict: Results per fold.
    """
    Results = {}
    original_dir = os.getcwd() 
    
    for fold, (train_idx, test_idx) in enumerate(cross_validation_indices):

        fold_name = f"fold{fold+1}"
        Results[fold_name] = {}

        fold_folder = os.path.join(data_folder, f"_fold{fold+1}") 
        os.makedirs(fold_folder, exist_ok=True)

        # Copy original feature name files
        for view in view_list:
            src = f"{data_folder}/{view}_featname.csv"
            dst = os.path.join(fold_folder, f"{view}_featname.csv")
            shutil.copy(src, dst)

        # Save labels
        np.savetxt(os.path.join(fold_folder, "labels_tr.csv"), labels[train_idx], fmt="%d", delimiter=",")
        np.savetxt(os.path.join(fold_folder, "labels_te.csv"), labels[test_idx], fmt="%d", delimiter=",")

        # Save omics data
        for i, view in enumerate(view_list):
            np.savetxt(os.path.join(fold_folder, f"{view}_tr.csv"), data_views[i].values[train_idx], delimiter=",")
            np.savetxt(os.path.join(fold_folder, f"{view}_te.csv"), data_views[i].values[test_idx], delimiter=",")

        # === Run MORE model with directory change ===
        
        try:
            os.chdir(repo_path)
            scores = generate_and_run_training_script(
                data_folder=fold_folder,
                view_list=view_list,
                script_name=script_path, 
                model_type = model_type, 
                dim_he_list = dim_he_list
            )

            # Parse accuracy, f1, roc_auc from stdout (2nd last line)
            Results[fold_name] = extract_metrics(scores.stdout)

        
        finally:
            os.chdir(original_dir)
    return Results 


def score_gene_rankings_vs_validation(ranked_lists, val_data, method_cutoff = 10, groundtruth_cutoff = 100):

    # compute TP, FP, FN

    metrics = {method: {
        'meanScore': None,
        'medianScore': None,
        'precision': None, 
        'recall': None, 
        'f1': None
    } for method in ranked_lists}


    for method, gene_list in ranked_lists.items():   
        subset = val_data.iloc[:groundtruth_cutoff, :].copy()
        val_subset = subset[subset['Entity'].isin(gene_list[:method_cutoff])]['Score']

        #val_subset = val_data.iloc[:groundtruth_cutoff,:][val_data['Entity'].isin(gene_list[:method_cutoff])]['Score'] 
        
        metrics[method]['meanScore'] = val_subset.mean()
        metrics[method]['medianScore'] = val_subset.median() 

        TP = val_subset.shape[0]  
        FP = method_cutoff - TP 
        FN = groundtruth_cutoff - TP  

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metrics[method]['TP'] = TP
        metrics[method]['FP'] = FP
        metrics[method]['FN'] = FN
        metrics[method]['precision'] = precision
        metrics[method]['recall'] = recall
        metrics[method]['f1'] = f1

    results = pd.DataFrame(metrics).T 
    results[['TP', 'FP', 'FN']]  = results[['TP', 'FP', 'FN']].astype(int)
    return results.sort_values(by = ['meanScore', 'medianScore'], ascending=[False, False]) 


def aggregate_cv_metrics(results_dict):
    import numpy as np

    # Initialize a dictionary to collect values for each metric
    metric_values = {}

    for fold_result in results_dict.values():
        for metric, value in fold_result.items():
            value = float(value)
            if metric not in metric_values:
                metric_values[metric] = []
            metric_values[metric].append(value)

    # Compute mean and std
    summary = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in metric_values.items()
    }

    return summary


def concat_scoring_across_cutoffs(
    ranked_lists,
    val_data,
    method_cutoffs=[10, 20, 50],
    groundtruth_cutoffs=[100]
):
    """

    """
    all_results = []

    for method_cutoff in method_cutoffs:
        for groundtruth_cutoff in groundtruth_cutoffs:
            # fix cases
            ranked_lists = {key:[value.lower() for value in values] for key, values in ranked_lists.items()} 
            val_data["Entity"] = val_data["Entity"].apply(lambda x: x.lower())
            result = score_gene_rankings_vs_validation(
                ranked_lists,
                val_data,
                method_cutoff=method_cutoff,
                groundtruth_cutoff=groundtruth_cutoff
            )
            result['method_cutoff'] = method_cutoff
            result['groundtruth_cutoff'] = groundtruth_cutoff
            all_results.append(result)

    return pd.concat(all_results).reset_index().rename(columns={'index': 'Method'})


def rank_features(weight_dict, ascending=True):
    """
    Return a list of features ranked by their weight.

    Args:
        weight_dict (dict): {feature: weight}
        ascending (bool): If True, lower weight = better (e.g., p-values).
                          If False, higher weight = better (e.g., SHAP values).

    Returns:
        list: List of feature names ranked from best to worst
    """
    sorted_items = sorted(weight_dict.items(), key=lambda x: x[1], reverse=not ascending)
    ranked_features = [feature for feature, _ in sorted_items]
    return ranked_features 



