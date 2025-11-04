import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

class Ttest_and_FoldChanger:
    """
    This class performs statistical analysis on numerical features in a given dataset.
    It includes:
    
    1. t-test and Log2 Fold Change Computation per Feature:
       - Computes p-values using an independent t-test (Welch's t-test) to compare features
         between a reference group and other groups.
       - Calculates log2 fold change for each feature by comparing the mean values of the reference group
         and the other group.

    2. Sample-Level Log2 Fold Change Computation:
       - Computes per-sample fold changes using the mean value of the reference group.
       - Converts fold changes into log2 values to standardize comparisons.
       
    Attributes:
        data (pd.DataFrame): The input dataset where the first column represents sample names,
            and the second column contains group labels (default column name: "Group").
        label_column_name (str): Column name representing the sample grouping (default: "Group").
        reference_group (str): The control/reference group for comparisons (default: "control").
        feature_type (str): The type of feature (i.e Metabolite, Gene, etc)
    """

    def __init__(self, data, label_column_name="Group", reference_group="control", feature_type = "Feature"):
        """
        Initializes the Ttest_and_FoldChanger class with the provided dataset and parameters.
        
        Args:
            data (pd.DataFrame): A dataframe where the first column contains sample names,
                and the second column contains group labels.
            label_column_name (str, optional): Name of the column containing group labels.
                Defaults to "Group".
            reference_group (str, optional): Name of the reference group. Defaults to "control".
            feature_type (str): The type of feature (i.e Metabolite, Gene, etc)
        """
        self.data = data.copy()
        self.label_column_name = label_column_name
        self.reference_group = reference_group
        self.feature_type = feature_type

    def calculate_ttest_and_log2fc(self):
        """
        Performs an independent t-test (Welch's t-test) for each numerical feature and computes log2 fold change.
        
        Returns:
            pd.DataFrame: A dataframe containing feature names, p-values, and log2 fold changes.
        """
        # Separate reference and other groups
        control_group = self.data[self.data[self.label_column_name] == self.reference_group]
        other_group = self.data[self.data[self.label_column_name] != self.reference_group]
        
        results = []
    
        # Iterate over all feature columns (excluding 'Sample ID' and 'Group')
        for feature in self.data.columns[2:]:  # Adjust index based on your column structure
            control_values = control_group[feature].dropna()
            other_values = other_group[feature].dropna()
            
            # Perform Welch’s t-test (does not assume equal variance)
            t_stat, p_value = ttest_ind(other_values, control_values, equal_var=False)
            
            # Compute log2 fold change
            mean_control = np.mean(control_values)
            mean_other = np.mean(other_values)
            log2fc = np.log2(mean_other / mean_control) if mean_control > 0 and mean_other > 0 else np.nan
            
            # Store results
            results.append({
                self.feature_type: feature,
                'p_value': p_value,
                'log2FoldChange': log2fc
            })
        
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df.sort_values(by = ["p_value"], ascending = [True])

    def compute_sample_level_foldchange(self):
        """
        Computes per-sample log2 fold changes using the mean values of the reference group.
        
        Returns:
            pd.DataFrame: A dataframe where each row represents a sample and each column represents
                          the log2 fold change of a feature.
        """
        df = self.data.copy()
        sample_name_column = df.columns[0]  # Extract sample identifier column name
        df.index = df.iloc[:, 0]  # Set sample names as index
        
        # Extract reference group
        reference_df = df[df[self.label_column_name] == self.reference_group]
        
        # Drop sample and group label columns
        df.drop(columns=df.columns[:2], inplace=True)
        reference_df.drop(columns=reference_df.columns[:2], inplace=True)

        # Compute mean values for reference group across all features
        reference = reference_df.mean(axis=0)
        
        # Compute fold changes per sample
        fold_changes = df.div(reference, axis=1)
        
        # Convert fold changes to log2 scale
        log2_fold_changes = np.log2(fold_changes + 1e-6)  # Small constant added to prevent log(0)
        log2_fold_changes.index.name = sample_name_column
        
        return log2_fold_changes.reset_index()


