import pandas as pd
import sys
import matplotlib.pyplot as plt
from modules.logger import logging
from modules.exception import CustomException

def feature_rank(top_features: dict = None, num_top_features: int = 20, feature_type: str = None, outdir: str = "outputs", save_feature_ranks:bool = True):
    """
    Ranks features based on their importance scores from different methods (e.g., SHAP, ANOVA), aggregates 
    the scores, and selects the top N features. The ranked features are saved to a CSV file.

    Args:
        top_features (dict): A dictionary containing feature importance scores from different methods.
        num_top_features (int): The number of top features to select and return.
        feature_type (str): The type of features being ranked (e.g., 'microRNA').
        outdir (str): The output directory where the ranked features CSV will be saved.
        save_feature_ranks: to save output in a file or not

    Returns:
        list: A list of the top N ranked features.
    """

    try:
        # Prepare ranking data 
        def rank_dict(d):
            # Create a list of keys sorted by their values in descending order
            sorted_keys = sorted(d, key=d.get, reverse=True)
            # Create a dictionary of ranks
            ranked_dict = {key: rank + 1 for rank, key in enumerate(sorted_keys)}
            return ranked_dict
    
        logging.info("Performing Feature Selection by Feature Ranking")
        # Apply ranking to each sub-dictionary
        ranked_data = {outer_key: rank_dict(outer_dict) for outer_key, outer_dict in top_features.items()}
        ranked_data_df = pd.DataFrame(ranked_data)
        
        # ANOVA features with many NaN values were filtered off due to high p-values, so we remove them
        ranked_data_df = ranked_data_df.dropna().reset_index().rename(columns={"index": feature_type})
        
        # Aggregate scores and sort by minimum
        ranked_data_df["overall score"] = ranked_data_df.iloc[:, 1:].sum(axis=1)
        ranked_data_df.sort_values(by="overall score", ascending=True, inplace=True)

        if save_feature_ranks:
            ranked_data_df.to_csv(f"{outdir}/ranked_features_df.csv", index=False)
        
        # Get top features and scores, sorted for uniform bar length decrease
        top_n_features = ranked_data_df.head(num_top_features)[feature_type].to_list()
        top_n_scores = ranked_data_df.head(num_top_features)["overall score"].to_list()

        # Normalize scores for uniform decrease in bar length
        max_score = max(top_n_scores)
        min_score = min(top_n_scores)
        normalized_scores = [max_score + min_score - score for score in top_n_scores]
        
        # Plot top N features by rank as vertical bars
        plt.figure(figsize=(12, 3))
        plt.bar(top_n_features, normalized_scores, color="skyblue")
        plt.xticks(rotation=45, ha="right", fontsize = 15)
        plt.title(f"Top {num_top_features} {feature_type} by Normalized Rank Score", fontsize = 15)
        plt.tight_layout()
        
        # Save the plot if needed
        if save_feature_ranks:
            plt.savefig(f"{outdir}/top_{num_top_features}_features_barplot.png", bbox_inches='tight')
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e, sys)
    return top_n_features








