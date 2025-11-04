from collections import defaultdict
import numpy as np
import pandas as pd

class ThresholdAggregator:
    """
    Implements the Threshold Algorithm (TA) for rank aggregation.

    This algorithm reads ranked feature lists row-by-row (depth-wise),
    computes a dynamic threshold at each step, and stops early when
    the top-k features have scores greater than or equal to the threshold.

    Attributes:
    - k (int or None): Number of features to return. If None, it is set to the mean length of the input lists. 

    Source: https://doi.org/10.1016/S0022-0000(03)00026-6
    """

    def __init__(self, k=None):
        """
        Initialize the aggregator.

        Parameters:
        - k (int or None): Number of features to return.
        """
        self.k = k

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
            sorted_items = sorted(col_scores.items(), key=lambda x: -x[1])  # descending by score
            ranked_lists.append(sorted_items)
    
        return ranked_lists
    
    def aggregate(self, ranked_lists):
        """
        Perform threshold-based aggregation.

        Parameters:
        - ranked_lists (list of lists): Each inner list contains tuples (feature, score),
                                        sorted in descending order of importance.

        Returns:
        - top_k_features (list): Aggregated top-k features based on the Threshold Algorithm.
        """

        if type(ranked_lists) != type([]):
            ranked_lists = self.rank_matrix_to_ranked_lists(ranked_lists)
            
        num_lists = len(ranked_lists)
        max_depth = max(len(lst) for lst in ranked_lists)

        if self.k is None:
            self.k = int(np.mean([len(lst) for lst in ranked_lists]))

        feature_scores = defaultdict(float)

        for depth in range(max_depth):
            threshold = 0

            for lst in ranked_lists:
                if depth < len(lst):
                    feature, score = lst[depth]
                    threshold += score
                    feature_scores[feature] += score

            # Get current top-k features
            top_k = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:self.k]
            top_k_scores = [score for _, score in top_k]

            # Early stopping condition
            if len(top_k_scores) == self.k and all(score >= threshold for score in top_k_scores):
                selected = [feature for feature, _ in top_k]
                return pd.DataFrame({"Name": selected,"Score": np.arange(len(selected) , 0, -1)/len(selected)})
                

        # Fallback: return best top-k features seen so far
        selected = [feature for feature, _ in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:self.k]]
        return pd.DataFrame({"Name": selected,"Score": np.arange(len(selected) , 0, -1)/len(selected)})