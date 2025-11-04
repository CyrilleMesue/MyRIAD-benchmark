import subprocess 
import os
import numpy as np
import pandas as pd

def generate_and_run_training_script(
    model_type: str,
    data_folder="ROSMAP",
    view_list=[1, 2, 3],
    script_name="temp_train_test.py",
    dim_he_list=None
):
    """
    Generates and runs a temporary training script for the specified model (MORE or MOGONET).

    Args:
        model_type (str): Either 'more' or 'mogonet' (case-insensitive).
        data_folder (str): Path to the dataset directory (e.g., "ROSMAP").
        view_list (list): List of integers representing the omics views to include.
        script_name (str): Name of the temporary Python script to generate and execute.
        dim_he_list (list or None): List of hidden embedding dimensions per view (MOGONET only).

    Returns:
        subprocess.CompletedProcess: Result of the script execution (stdout, stderr).
    
    Raises:
        ValueError: If an unsupported model_type is passed.
        subprocess.CalledProcessError: If the script execution fails.
    """
    model_type = model_type.lower()
    num_class = 2

    if model_type == "more":
        script_content = f"""
from train_test import train_test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":    
    num_epoch_pretrain = 500
    num_epoch = 1500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    train_test("{data_folder}", {view_list}, {num_class},
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch) 
"""
    elif model_type == "mogonet":
        if dim_he_list is None:
            raise ValueError("dim_he_list must be provided for MOGONET.")
        script_content = f"""
from train_test import train_test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":    
    num_epoch_pretrain = 500
    num_epoch = 1500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    train_test("{data_folder}", {view_list}, {num_class},
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, {dim_he_list}) 
"""
    else:
        raise ValueError("Unsupported model_type. Choose either 'more' or 'mogonet'.")

    # Write and run
    with open(script_name, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(["python", script_name], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("❌ Script failed with error:\n", e.stderr)
        raise

    return result


def generate_and_run_feat_imp_script(
    data_folder="ROSMAP",
    view_list=[1, 2, 3],
    script_name="run_feat_importance.py", feature_rankings_path='more-feature-ranks.csv'
):
    """
    Generates and executes a temporary script for computing feature importance using MORE.

    Parameters:
        data_folder (str): Dataset folder (e.g., 'ROSMAP' or 'BRCA').
        view_list (list): List of view indices to evaluate.
        script_name (str): Filename for the generated script.
    """
    num_class = 2

    script_content = f"""
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = '{data_folder}'
    model_folder = os.path.join(data_folder, 'models')
    view_list = {view_list}
    num_class = {num_class}

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep+1)), 
                                    view_list, num_class)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(featimp_list_list, feature_rankings_path="{feature_rankings_path}")
"""

    # Save script to disk
    with open(script_name, "w") as f:
        f.write(script_content)

    # Run the generated script
    print("🚀 Running the script...")
    try:
        subprocess.run(["python", script_name], check=True) 
    except subprocess.CalledProcessError as e:
        print("❌ Script failed with error:\n", e.stderr)
        raise


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
