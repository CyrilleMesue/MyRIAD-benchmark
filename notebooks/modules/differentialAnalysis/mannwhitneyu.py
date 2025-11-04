import pandas as pd
import numpy as np
import sys
from scipy.stats import mannwhitneyu
from modules.logger import logging
from modules.exception import CustomException


class MannWhitneySelector:
    """
    Mann–Whitney U Test (Wilcoxon rank-sum test) Feature Scorer.

    Performs a non-parametric test to assess whether feature distributions differ
    between two independent groups.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels_column: str = "Diagnosis"
    ):
        """
        Initialize the selector.

        Parameters:
        - data (pd.DataFrame): DataFrame including sample IDs, labels, and features
        - labels_column (str): Column name containing group labels
        """
        self.data = data
        self.labels_column = labels_column

    def fit(self) -> dict:
        """
        Perform the Mann–Whitney U test for all features.

        Returns:
        - feature_pval_map (dict): {feature_name: p-value}
        """
        try:
            logging.info("Performing Mann–Whitney U Test for all features.")

            df = self.data.copy()
            feature_cols = df.columns.difference([self.labels_column])
            labels = df[self.labels_column].unique()

            if len(labels) != 2:
                raise CustomException("Mann–Whitney U test requires exactly two groups.")

            group1, group2 = labels
            group1_df = df[df[self.labels_column] == group1]
            group2_df = df[df[self.labels_column] == group2]

            feature_pval_map = {}

            for feature in feature_cols:
                values1 = group1_df[feature].dropna()
                values2 = group2_df[feature].dropna()

                if len(values1) < 2 or len(values2) < 2:
                    p_val = np.nan  # Not enough data
                else:
                    _, p_val = mannwhitneyu(values1, values2, alternative='two-sided')

                feature_pval_map[feature] = p_val

            logging.info("Mann–Whitney U Test completed.")
            return feature_pval_map

        except Exception as e:
            logging.error("Mann–Whitney test failed.")
            raise CustomException(e, sys)
