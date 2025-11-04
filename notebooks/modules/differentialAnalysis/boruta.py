import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from modules.logger import logging
from modules.exception import CustomException


class BorutaSelector:
    """
    Boruta Feature Selection using RandomForestClassifier.

    Accepts a DataFrame with sample IDs, labels, and features.
    Returns dictionaries mapping features to their ranking and selection status.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels_column: str = "Diagnosis",
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Boruta feature selector.

        Parameters:
        - data (pd.DataFrame): DataFrame including sample IDs, labels, and features
        - labels_column (str): Column name containing group labels
        - max_depth (int): Max depth of RandomForestClassifier
        - random_state (int): Seed for reproducibility
        """
        self.data = data
        self.labels_column = labels_column
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self) -> tuple[dict, dict]:
        """
        Run Boruta feature selection.

        Returns:
        - feature_rank_map (dict): {feature_name: rank (1 = selected, >1 = rejected)}
        - feature_support_map (dict): {feature_name: 1 if selected, 0 if not}
        """
        try:
            # Extract features automatically
            df = self.data.copy()
            feature_cols = df.columns.difference([self.labels_column])
            X = df[feature_cols].values
            y = df[self.labels_column].values.ravel()

            # Define RandomForest and Boruta
            rf = RandomForestClassifier(
                n_jobs=-1,
                class_weight='balanced',
                max_depth=self.max_depth,
                random_state=self.random_state
            )

            feat_selector = BorutaPy(
                estimator=rf,
                n_estimators='auto',
                verbose=0,
                random_state=self.random_state
            )

            logging.info("Fitting Boruta selector...")
            feat_selector.fit(X, y)

            # Build output maps
            feature_rank_map = dict(zip(feature_cols, feat_selector.ranking_))
            feature_support_map = dict(zip(feature_cols, feat_selector.support_.astype(int)))

            logging.info(f"Boruta finished. {sum(feat_selector.support_)} features selected.")
            return feature_rank_map, feature_support_map

        except Exception as e:
            logging.error("Boruta feature selection failed.")
            raise CustomException(e, sys)
