from collections import defaultdict
import numpy as np
import pandas as pd

class MedRankAggregator:
    """
    Implements the MedRank Algorithm (MA) for feature aggregation.

    Features are scanned row-by-row from each ranked list. A feature is added to the result
    once it has appeared in at least a certain proportion (threshold) of the lists.
    Aggregation stops once k features are selected.

    Parameters:
    - k (int or None): Number of features to return. If None, it's set to the average list length.
    - min_fraction (float): Minimum fraction of lists a feature must appear in (e.g., 0.2 for 20%)
    """

    def __init__(self, k=None, min_fraction=0.2):
        self.k = k
        self.min_fraction = min_fraction

    def rank_matrix_to_ranked_lists(self, rank_matrix):
        """
        Converts a pandas rank matrix to a list of ranked feature-score lists per column.
    
        Parameters:
        - rank_matrix (pd.DataFrame): Rows are features, columns are ranked lists (with numeric scores)
    
        Returns:
        - list of lists: Each inner list contains (feature, score) tuples sorted by descending score
        """

        if not isinstance(rank_matrix, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame for `rank_matrix`.")
            
        ranked_lists = []
    
        for col in rank_matrix.columns:
            col_scores = rank_matrix[col].dropna()
            sorted_items = [item[0] for item in sorted(col_scores.items(), key=lambda x: x[1])]  # ascending by score
            ranked_lists.append(sorted_items)
    
        return ranked_lists
        
    def aggregate(self, ranked_lists):
        """
        Perform MedRank aggregation.

        Parameters:
        - ranked_lists (list of lists): Each inner list contains ranked feature names (strings)

        Returns:
        - selected_features (list): Aggregated list of top-k features based on occurrence frequency
        """ 

        if type(ranked_lists) != type([]):
            ranked_lists = self.rank_matrix_to_ranked_lists(ranked_lists)

        num_lists = len(ranked_lists)
        max_depth = max(len(lst) for lst in ranked_lists)

        # Automatically determine k if not provided
        if self.k is None:
            self.k = int(np.mean([len(lst) for lst in ranked_lists]))

        min_count = int(np.ceil(self.min_fraction * num_lists))
        feature_counts = defaultdict(int)
        selected = []

        for depth in range(max_depth):
            for lst in ranked_lists:
                if depth < len(lst):
                    feature = lst[depth]
                    feature_counts[feature] += 1

                    if (
                        feature_counts[feature] >= min_count
                        and feature not in selected
                    ):
                        selected.append(feature)

                        if len(selected) >= self.k:
                            return pd.DataFrame({"Name": selected,"Score": np.arange(1, len(selected) + 1)/len(selected)})

        # If we still don't have enough, fill remaining spots with most frequent unseen features
        if len(selected) < self.k:
            remaining = sorted(
                ((feat, count) for feat, count in feature_counts.items() if feat not in selected),
                key=lambda x: -x[1]
            )
            for feat, _ in remaining:
                selected.append(feat)
                if len(selected) >= self.k:
                    break
                    
        return pd.DataFrame({"Name": selected,"Score": np.arange(1, len(selected) + 1)/len(selected)})
