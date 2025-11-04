import pandas as pd
import subprocess
import tempfile
import os
import sys
import shutil
from sklearn import set_config

from modules.logger import logging
from modules.exception import CustomException

set_config(transform_output="pandas")


class LIMMA:
    """
    LIMMA Differential Expression Analysis (via R)

    This class wraps the R limma pipeline using subprocess to run an external R script.
    It expects a tidy DataFrame containing expression values, sample IDs, and labels.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels_column: str = "Diagnosis",
        reference_class: str = "Control",
        sample_id_column: str = "Sample ID",
        feature_type: str = "MicroRNA",
        r_script_path: str = "./RScripts/run_limma.R"
    ):
        """
        Initialize the LIMMA runner.

        Parameters:
        - data (pd.DataFrame): Long-format expression data including sample IDs and labels
        - labels_column (str): Name of the column containing group labels
        - reference_class (str): The reference group (e.g., "Control")
        - sample_id_column (str): Name of the column containing sample IDs
        - feature_type (str): Name of the feature identifier column (e.g., "Gene")
        - r_script_path (str): Path to the R script
        """
        self.data = data
        self.labels_column = labels_column
        self.reference_class = reference_class
        self.sample_id_column = sample_id_column
        self.feature_type = feature_type
        self.r_script_path = r_script_path

    def fit(self) -> tuple[dict, dict]:
        """
        Run LIMMA and return differential expression results.

        Returns:
        - logFC_dict (dict): {feature: log fold change}
        - padj_dict (dict): {feature: adjusted p-value}
        """
        temp_dir = tempfile.mkdtemp(prefix="limma_tmp_")

        try:
            expression_path = os.path.join(temp_dir, "expression.csv")
            metadata_path = os.path.join(temp_dir, "metadata.csv")
            result_path = os.path.join(temp_dir, "results.csv")

            # Prepare expression matrix
            df = self.data.copy()
            sample_ids = df[self.sample_id_column]
            X = df.drop(columns=[self.sample_id_column, self.labels_column])
            features = X.columns
            X = X.T
            X.columns = sample_ids
            X.insert(0, self.feature_type, features)
            X.to_csv(expression_path, index=False)

            # Prepare metadata
            y_df = df[[self.sample_id_column, self.labels_column]]
            y_df.to_csv(metadata_path, index=False)

            # Run the R script
            logging.info(f"Running LIMMA via R script: {self.r_script_path}")
            subprocess.run([
                "Rscript",
                self.r_script_path,
                expression_path,
                metadata_path,
                self.labels_column,
                result_path,
                self.reference_class
            ], check=True)

            # Read result
            results = pd.read_csv(result_path, index_col=0)
            if "logFC" not in results.columns or "adj.P.Val" not in results.columns:
                raise CustomException("Missing expected columns ('logFC', 'adj.P.Val') in LIMMA result.")

            logfc_dict = dict(zip(results.index, results["logFC"]))
            padj_dict = dict(zip(results.index, results["adj.P.Val"]))

            logging.info(f"LIMMA completed successfully. {len(logfc_dict)} features returned.")
            return logfc_dict, padj_dict

        except Exception as e:
            logging.error("LIMMA failed.")
            raise CustomException(e, sys)

        finally:
            shutil.rmtree(temp_dir)
