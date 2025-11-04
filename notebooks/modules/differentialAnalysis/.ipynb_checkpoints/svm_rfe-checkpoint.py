import pandas as pd
import numpy as np
import sys
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from modules.logger import logging
from modules.exception import CustomException


class SVMRFESelector:
    """
    Feature selection using SVM-RFE (Support Vector Machine + Recursive Feature Elimination).

    Returns:
        - Feature ranks (1 = most important)
        - Binary selection mask (1 = selected, 0 = not selected)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels_column: str = "Diagnosis",
        n_features_to_select: int = None,
        kernel: str = "linear",
        random_state: int = 42
    ):
        """
        Parameters:
        - data (pd.DataFrame): Full dataset including labels and features
        - labels_column (str): Name of the target/label column
        - n_features_to_select (int): Number of top features to select
        - kernel (str): SVM kernel (must be 'linear' for RFE to work correctly)
        - random_state (int): Random seed
        """
        self.data = data
        self.labels_column = labels_column
        self.n_features_to_select = n_features_to_select
        self.kernel = kernel
        self.random_state = random_state

    def fit(self) -> tuple[dict, dict]:
        """
        Perform SVM-RFE and return feature ranks and selection mask.

        Returns:
        - feature_rank_map (dict): {feature_name: rank}
        - feature_support_map (dict): {feature_name: 1 if selected, 0 otherwise}
        """
        try:
            logging.info("Performing SVM-RFE Feature Selection...")

            df = self.data.copy()
            X = df.drop(columns=[self.labels_column])
            y = df[self.labels_column].values.ravel()
            feature_names = X.columns

            # Use linear SVC for SVM-RFE
            estimator = SVC(kernel=self.kernel, random_state=self.random_state)
            selector = RFE(estimator, n_features_to_select=self.n_features_to_select)
            selector.fit(X, y)

            # Build result maps
            feature_rank_map = dict(zip(feature_names, selector.ranking_))
            feature_support_map = dict(zip(feature_names, selector.support_.astype(int)))

            logging.info(f"SVM-RFE completed. {sum(selector.support_)} features selected.")
            return feature_rank_map, feature_support_map

        except Exception as e:
            logging.error("SVM-RFE failed.")
            raise CustomException(e, sys)
