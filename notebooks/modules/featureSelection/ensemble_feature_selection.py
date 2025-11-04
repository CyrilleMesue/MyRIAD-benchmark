import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from modules.logger import logging
from modules.exception import CustomException 
from modules.verbose_config import BaseConfigurable
from modules.featureSelection.RRA import RobustRankAggregator 
from modules.featureSelection.MRA import MedRankAggregator
from modules.featureSelection.TA import ThresholdAggregator

class EnsembleFeatureRanker(BaseConfigurable):
    """
    Implements ensemble feature ranking strategies for robust feature selection using different methods
    (e.g., SHAP, LIME, ANOVA, t-test).

    Attributes:
        feature_weights (dict): Dictionary containing feature importance scores from different methods.
        feature_type (str): A label indicating the type of features (e.g., gene, miRNA).
        outdir (str): Output directory for saving results.
        save_feature_ranks (bool): Whether to save ranked features and plots.
        num_top_features (int): Number of top features to visualize and return.
        decreasing_importance_cases (list): List of methods where lower scores imply higher importance (e.g., p-values).
    """

    def __init__(self, 
                 feature_weights: dict = None,
                 feature_type: str = None,
                 outdir: str = None,
                 num_top_features: int = 20,
                 decreasing_importance_cases: list = ["t_test"],
                 save_results:bool = True,
                 show_plots:bool = True
                ):
        """
        Initializes the EnsembleFeatureRanking class.

        Args:
            feature_weights (dict): Feature importance scores from various methods.
            feature_type (str): Type of feature (e.g., gene).
            outdir (str): Directory to save outputs.
            save_feature_ranks (bool): Flag to save results.
            num_top_features (int): Number of top features to report.
            decreasing_importance_cases (list): List of methods where smaller values are better (e.g., p-values).
        """
        self.feature_weights = feature_weights
        self.feature_type = feature_type
        self.outdir = outdir
        self.num_top_features = num_top_features
        self.decreasing_importance_cases = decreasing_importance_cases 
        self.save_results = save_results
        self.show_plots = show_plots 

    def stats_method(self, weight_matrix, method):
        """
        Aggregate feature importance scores across multiple selectors using a statistical method.
    
        Parameters:
        ----------
        weight_matrix : pd.DataFrame
            A DataFrame where rows are features and columns represent scores from different feature selectors.
        
        method : str
            The aggregation method to apply. Supported methods:
            - 'mean'       : Arithmetic mean of scores across selectors.
            - 'max'        : Maximum score across selectors.
            - 'median'     : Median of scores across selectors.
            - 'geom.mean'  : Geometric mean of scores (requires all values to be positive).
    
        Returns:
        -------
        pd.DataFrame
            A DataFrame with columns:
            - 'Name'  : Feature names (from index of weight_matrix)
            - 'Score' : Aggregated score computed using the specified method,
                        sorted in descending order.
    
        Raises:
        ------
        ValueError:
            If an unsupported method is provided.
        """
    
        if method == 'mean':
            aggregated_ranks = pd.DataFrame({"Name":weight_matrix.index, "Score": weight_matrix.mean(axis=1)}) 
        elif method == 'max':
            aggregated_ranks = pd.DataFrame({"Name":weight_matrix.index, "Score": weight_matrix.max(axis=1)})
        elif method == 'median':
            aggregated_ranks = pd.DataFrame({"Name":weight_matrix.index, "Score": weight_matrix.median(axis=1)})
        elif method == 'geom.mean':
            aggregated_ranks = pd.DataFrame({"Name":weight_matrix.index, "Score": weight_matrix.apply(
                lambda row: np.exp(np.nanmean(np.log(row+1e-8))), axis=1
            )})
            
        else:
            print('Wrong methods Provided')
    
        return aggregated_ranks.sort_values(by = "Score", ascending = False)

    def Rank_by_Ranks(self, targeted_features: list = [], plot: bool = False, normalize_ranks:bool = True, method:str="mean"):
        """
        Ranks features by averaging their ranks from different feature selection methods.

        Args:
            targeted_features (list): Specific features to include in ranking.
            plot (bool): Whether to plot the top N features.

        Returns:
            dict: Dictionary of features and their aggregated rank scores.
        """
        try:
            # Select only targeted features if provided
            if len(targeted_features) != 0:
                feature_weights = {method: {feature: weights[feature] for feature in targeted_features}  for method, weights in self.feature_weights.items()}
            else:
                feature_weights = self.feature_weights.copy()

            # Function to rank a dictionary
            def rank_dict(d, reverse=True):
                sorted_keys = sorted(d, key=d.get, reverse=reverse)
                return {key: rank + 1 for rank, key in enumerate(sorted_keys)}

            logging.info("Performing Feature Selection by Aggregating Feature Ranks")

            # Apply ranking to each feature selection method
            ranked_data = {method: rank_dict(scores) for method, scores in feature_weights.items()}

            # Reverse ranks for decreasing importance cases (e.g., p-values)
            for case in self.decreasing_importance_cases:
                if case in feature_weights:
                    ranked_data[case] = rank_dict(feature_weights[case], reverse=False)

            ranked_data_df = pd.DataFrame(ranked_data).dropna().reset_index().rename(columns={"index": self.feature_type})
            ranked_data_df.index = ranked_data_df[self.feature_type] 
            ranked_data_df.drop(columns = [self.feature_type], inplace = True)

            if method == "mra":
                mra = MedRankAggregator(k=ranked_data_df.shape[0], min_fraction=0.25)
                aggregated_ranks = mra.aggregate(ranked_lists = ranked_data_df) 
            else:
                robust_rank_aggregator = RobustRankAggregator(method=method, normalize_ranks=normalize_ranks)
                aggregated_ranks = robust_rank_aggregator.aggregate_ranks(rmat=ranked_data_df)
            aggregated_ranks.rename(columns = {"Name": self.feature_type, "Score":f"{method.upper()}_Rank_Score"}, inplace = True)
            aggregated_ranks.sort_values(by=f"{method.upper()}_Rank_Score", ascending=True, inplace=True)

            # save results
            self.maybe_save(aggregated_ranks, f"{self.outdir}/Features_Ranked_by_{method.upper()}_Rank_Score.csv", 
                       object_type = "table", data_name = f"Ranks_by_{method.upper()}_Rank_Score")

            top_n_features = aggregated_ranks.head(self.num_top_features)[self.feature_type].to_list()
            top_n_scores = aggregated_ranks.head(self.num_top_features)[f"{method.upper()}_Rank_Score"].to_list()

            if plot:
                max_score = max(top_n_scores)
                min_score = min(top_n_scores)
                normalized_scores = [max_score + min_score - score for score in top_n_scores]

                plt.figure(figsize=(12, 3))
                plt.bar(top_n_features, normalized_scores, color="skyblue")
                plt.xticks(rotation=45, ha="right", fontsize=15)
                plt.title(f"Top {self.num_top_features} {self.feature_type} by Normalized {method.upper()}_Rank_Score", fontsize=15)
                plt.tight_layout()

                # save plots
                self.maybe_save(plt.gcf(), f"{self.outdir}/top_{self.num_top_features}_features_ranked_by_{method.upper()}_Rank_Score_barplot.png", 
                           object_type = "figure", data_name = "Figure")
                
                # Show figure
                self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

        return dict(zip(aggregated_ranks[self.feature_type], aggregated_ranks[f"{method.upper()}_Rank_Score"]))

    def Rank_by_Weights(self, targeted_features: list = [], plot: bool = True,scale_weights: bool = True, method:str = 'mean'):
        """
        Ranks features by averaging their raw importance scores (weights) across methods.

        Args:
            scale_weights (bool): If True, normalizes feature weights between 0 and 1 for each row.
            plot (bool): Whether to plot the top N features.
            targeted_features (list): Subset of features to rank.
            methods (str): Whether to take the mean, max or median weight.

        Returns:
            dict: Dictionary of features and their aggregated weight scores.
        """

        try:

            # Function to reverse weights a dictionary:for features like pvalues
            def reverse_weight_dict(d):
                max_ = max(d.values())
                return {key: max_ - value for key, value in d.items()}
                
            if len(targeted_features) != 0:
                feature_weights = {feature: self.feature_weights[feature] for feature in targeted_features}
            else:
                feature_weights = self.feature_weights.copy()

            logging.info("Performing Feature Selection by Aggregating Feature Weights")

            # Reverse ranks for decreasing importance cases (e.g., p-values)
            for case in self.decreasing_importance_cases:
                if case in feature_weights:
                    feature_weights[case] = reverse_weight_dict(feature_weights[case])

            feature_weights_df = pd.DataFrame(feature_weights).dropna().reset_index().rename(columns={"index": self.feature_type})
            feature_weights_df.index = feature_weights_df[self.feature_type] 
            feature_weights_df.drop(columns = [self.feature_type], inplace = True)

            if scale_weights:
                feature_weights_df = (feature_weights_df - feature_weights_df.min(axis=0)) \
                     / (feature_weights_df.max(axis=0) - feature_weights_df.min(axis=0))

            if method == "ta":
                ta = ThresholdAggregator(k= feature_weights_df.shape[0])
                aggregated_weights = ta.aggregate(ranked_lists = feature_weights_df)
            else:
                aggregated_weights = self.stats_method(weight_matrix = feature_weights_df, method=method)

            aggregated_weights.rename(columns = {"Name": self.feature_type, "Score":f"{method.upper()}_Weight_Score"}, inplace = True)
            aggregated_weights.sort_values(by=f"{method.upper()}_Weight_Score", ascending=False, inplace=True)

            # save results
            add = "_scaled" if scale_weights else ""
            self.maybe_save(aggregated_weights, f"{self.outdir}/Features_Ranked_by_{method.upper()}_Weight_Score{add}.csv", 
                       object_type = "table", data_name = f"Rank_by_{method.upper()}_Weight_Score")

            top_n_features = aggregated_weights.head(self.num_top_features)[self.feature_type].to_list()
            top_n_scores = aggregated_weights.head(self.num_top_features)[f"{method.upper()}_Weight_Score"].to_list()

            if plot:
                max_score = max(top_n_scores)
                min_score = min(top_n_scores)
                normalized_scores = [max_score + min_score - score for score in top_n_scores]

                plt.figure(figsize=(12, 3))
                plt.bar(top_n_features, top_n_scores, color="skyblue")
                plt.xticks(rotation=45, ha="right", fontsize=15)
                if scale_weights:
                    plt.title(f"Top {self.num_top_features} {self.feature_type}s by {method.upper()} Weight Score",fontsize=15)
                else:
                    plt.title(f"Top {self.num_top_features} {self.feature_type}s by {method.upper()} Weight Score",fontsize=15)
                plt.tight_layout()

                # save plots
                self.maybe_save(plt.gcf(), f"{self.outdir}/top_{self.num_top_features}_features_ranked_by_{method.upper()}_Weight_Score{add}_barplot.png", 
                           object_type = "figure", data_name = "Figure")
                # Show figure
                self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

        return dict(zip(aggregated_weights[self.feature_type], aggregated_weights[f"{method.upper()}_Weight_Score"]))

    def select_by_Frequency_of_Occurence_from_topN(self, N: int = 30):
        """
        Selects features that frequently occur in the top N of each method.

        Args:
            N (int): Number of top features to consider from each method.

        Returns:
            set: Features that are consistently ranked in the top N across methods.
        """
        try:
            def rank_dict(d, reverse=True):
                sorted_keys = sorted(d, key=d.get, reverse=reverse)
                return {key: rank + 1 for rank, key in enumerate(sorted_keys)}

            logging.info("Selecting Features by Frequency of Occurrence")

            feature_weights = self.feature_weights.copy()
            ranked_data = {method: rank_dict(scores) for method, scores in feature_weights.items()}
 
            # Reverse ranks for decreasing importance cases (e.g., p-values)
            for case in self.decreasing_importance_cases:
                if case in self.feature_weights:
                    ranked_data[case] = rank_dict(self.feature_weights[case], reverse=False)

            ranked_data_df = pd.DataFrame(ranked_data).reset_index().rename(columns={"index": self.feature_type})

            selection_space = {}
            for method in ranked_data_df.columns[1:]:
                ranked_data_df.sort_values(by=method, inplace=True)
                selection_space[method] = set(ranked_data_df.iloc[:N, 0].to_list())

            selected_features = set.intersection(*selection_space.values())

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

        return selected_features
