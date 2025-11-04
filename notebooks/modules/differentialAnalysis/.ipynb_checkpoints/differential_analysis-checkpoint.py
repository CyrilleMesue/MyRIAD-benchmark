# Load general packages
import pandas as pd
import numpy as np
import os
import sys
from tqdm.notebook import tqdm

# Load sklearn packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn import set_config

# Load visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Load statistical packages
from scipy.stats import rankdata, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load specialized packages
import shap
import lime
import lime.lime_tabular

# Load Custom Modules
from modules.utils import save_json, load_json
from modules.differentialAnalysis.shap_analysis import SHAP_Analysis
from modules.differentialAnalysis.lime_analysis import LIME_Analysis
from modules.differentialAnalysis.feature_importance_analysis import FeatureImportance_Analysis
from modules.differentialAnalysis.cox_analysis import CoxFeatureRanker 
from modules.differentialAnalysis.limma import LIMMA
from modules.differentialAnalysis.boruta import BorutaSelector
from modules.differentialAnalysis.mannwhitneyu import MannWhitneySelector
from modules.differentialAnalysis.svm_rfe import SVMRFESelector
from modules.logger import logging
from modules.exception import CustomException
from modules.verbose_config import BaseConfigurable

# Set sklearn configuration
set_config(transform_output="pandas")


class DifferentiatingFactorAnalysis(BaseConfigurable):
    """
    The DifferentiatingFactorAnalysis class is designed to perform various statistical and machine learning analyses
    to identify the features that differentiate between different classes or conditions in a dataset. 

    This class includes methods for the following analyses:
    
    - SHAP Analysis: Uses SHAP (SHapley Additive exPlanations) to interpret the output of machine learning models. In this class, a shap summary plot, cluster plot and feature importances are generated.
    - Lime Analysis: Utilizes LIME (Local Interpretable Model-agnostic Explanations) to provide local explanations for model predictions. A lime summary plot and feature importances are generated.
    - ANOVA: Conducts Analysis of Variance (ANOVA) to identify features that have significant differences between groups.
    - t-test: Performs t-tests to compare the means of two groups and identify significant differences.
    - Feature Importance Analysis: Evaluates the importance of features in a predictive model to determine which features have the most influence on the outcome.

    Methods
    -------
    perform_shap_analysis():
        Performs SHAP analysis on the provided model and dataset.
    
    perform_lime_analysis(y):
        Performs LIME analysis on the provided model and dataset.
    
    perform_anova():
        Conducts ANOVA to find features with significant differences between groups.
    
    perform_t_test():
        Performs t-tests to identify significant differences between two groups.
    
    perform_randomforest_feature_importance_analysis():
        Computes Random Forest Feature Importances.

    perform_xgb_feature_importance_analysis():
        Computes XGBoost Feature Importances.

    perform_permutation_feature_importance_analysis():
        Computes Permutation Feature Importances Using Random Forest Model.

    run_all_analyses():
        Run All Methods Listed from the analyses parameter.


    Attributes
    ----------
    
    X : DataFrame or array-like
        The feature matrix.
    
    y : Series or array-like
        The target variable.
    """

    
    def __init__(self, 
                 data=None, 
                 analyses=None, 
                 labels_column="Diagnosis", 
                 reference_class:str = "Control",
                 sample_id_column="Sample ID", 
                 outdir: str = "output", 
                 mode: str = "classification",
                 feature_type: str = "microRNA",
                 test_size: float = 0.2,
                 lime_global_explanation_sample_num:int =10,
                 shap_model_finetune:bool = False,
                 lime_model_finetune:bool = False,
                 cox_model_finetune:bool = False,
                 n_folds:int = 5,
                 scoring:str = "f1",
                 top_num_features_to_plot:int = 20,
                 featureimportance_finetune:bool = False,
                 verbose:bool = True,
                 save_results:bool = True,
                 show_plots:bool = True
                ):
        """
        Initializes the DifferentiatingFactorAnalysis class with the specified parameters.

        Parameters:
                data (pd.DataFrame, optional): The input data as a pandas DataFrame.
                analyses (list of str, optional): List of analyses to perform. If None, a predefined list is used.
                labels_column (str, optional): Column name for labels. Default is "Diagnosis".
                reference_class (str, optional): The reference class for comparison. Default is "Control".
                sample_id_column (str, optional): Column name for sample IDs. Default is "Sample ID".
                outdir (str, optional): Output directory for saving results. Default is "output".
                mode (str, optional): Mode of operation, e.g., "classification". Default is "classification".
                feature_type (str, optional): Type of feature, e.g., "microRNA". Default is "microRNA".
                test_size (float, optional): Proportion of data to be used as the test set. Default is 0.2.
                lime_global_explanation_sample_num (int, optional): Number of samples to use for global LIME explanations. Default is 10.
                shap_model_finetune (bool, optional): Flag to indicate if model fine-tuning is applied for SHAP analysis. Default is False.
                lime_model_finetune (bool, optional): Flag to indicate if model fine-tuning is applied for LIME analysis. Default is False.
                n_folds (int, optional): Number of folds for cross-validation. Default is 5.
                scoring (str, optional): Scoring method to evaluate model performance. Default is "f1".
                top_num_features_to_plot (int, optional): Number of top features to display in feature importance plots. Default is 20.
                featureimportance_finetune (bool, optional): Flag to indicate if fine-tuning is applied for feature importance analysis. Default is True.


        Attributes:
        X (pd.DataFrame): Features extracted from the data.
        y (np.ndarray): Encoded labels.
        labels (pd.Series): Original labels.
        label_encodings (dict): Dictionary mapping encoded labels to original labels.
        class_names (list): List of unique class names.
        outdir (str): Output directory.
        mode (str): Mode of operation.
        feature_type (str): Type of feature.
        test_size (float): Test set size.
        SHAP_Analyzer: Placeholder for SHAP analysis object.
        LIME_Analyzer: Placeholder for LIME analysis object.
        ANOVA_Analyzer: Placeholder for ANOVA analysis object.
        feature_IMPORT_RF_Analyzer: Placeholder for Random Forest feature importance analysis object.
        feature_IMPORT_XGB_Analyzer: Placeholder for XGBoost feature importance analysis object.
        feature_IMPORT_PERMUTATION_Analyzer: Placeholder for permutation feature importance analysis object.
        t_TEST_Analyzer: Placeholder for t-test analysis object.
        analyses (list of str): List of analyses to perform.
        feature_weights (dict): Dictionary for storing top features for each analysis.
        """

        # store variables
        sns.set_theme(style="darkgrid")
        self.data = data
        self.X = data.drop([labels_column, sample_id_column], axis=1)
        self.y = LabelEncoder().fit_transform(data[labels_column])
        self.feature_map = {feature: f"Feature_{i}" for i, feature in enumerate(self.X.columns)}
        self.feature_map_reverse = {value:key for key,value in self.feature_map.items()}
        self.X.columns = list(map(lambda x: self.feature_map[x], self.X.columns)) # we map features as certaim may contain unsupported chars {/,\,%, etc}
        self.labels = data[labels_column]
        self.label_encodings = dict(zip(self.y, self.labels))
        self.class_names = list(self.labels.unique())
        self.outdir = outdir
        self.mode = mode
        self.reference_class = reference_class
        self.labels_column = labels_column
        self.sample_id_column = sample_id_column
        self.feature_type = feature_type
        self.test_size = test_size
        self.SHAP_Analyzer = None
        self.LIME_Analyzer = None
        self.ANOVA_Analyzer = None
        self.feature_IMPORT_RF_Analyzer = None
        self.feature_IMPORT_XGB_Analyzer = None
        self.feature_IMPORT_PERMUTATION_Analyzer = None
        self.t_TEST_Analyzer = None
        self.analyses = ["shap", "lime", "anova", "rf_feature_importance",  "xgb_feature_importance",  "rf_permutation_feature_importance",  "xgb_permutation_feature_importance", "t_test", "lasso", "ridge", "elasticnet", "limma", "boruta", "mannwhitneyu", "svm_rfe"] \
        if analyses is None else analyses
        self.n_folds = n_folds
        self.top_num_features_to_plot = top_num_features_to_plot
        self.fitted_model_rf = None
        self.fitted_model_xgb = None 
        self.save_results = save_results
        self.show_plots = show_plots
        self.verbose = verbose
        
        # select random samples from each class to compare
        np.random.seed(42) # Ensure reproducibility by setting a random seed
        
        # Filter indices for AD and Control samples
        class0_indices = data[data[labels_column] == self.class_names[0]].index
        class1_indices = data[data[labels_column] == self.class_names[1]].index
        
        # Randomly select one index from each group
        random_class0_index = np.random.choice(class0_indices)
        random_class1_index = np.random.choice(class1_indices)
        self.random_samples = {self.class_names[0]:random_class0_index , 
                               self.class_names[1]:random_class1_index }

        # Initialize Analyses
        if "shap" in self.analyses:
            self.SHAP_Analyzer = SHAP_Analysis(
                                 X=self.X, 
                                 y=self.y, 
                                 outdir = f"{self.outdir}/shap", 
                                 random_samples = self.random_samples,
                                 feature_type = self.feature_type, 
                                 feature_map_reverse = self.feature_map_reverse, 
                                 model_finetune=shap_model_finetune, 
                                 fine_tune_cv_nfolds = n_folds, 
                                 scoring = scoring, 
                                 top_num_features_to_plot = top_num_features_to_plot,
                                 save_results = save_results,
                                 show_plots = show_plots
                                 )
            
        if "lime" in self.analyses:
            self.LIME_Analyzer = LIME_Analysis(
                                 X=self.X, 
                                 y=self.y, 
                                 outdir = f"{self.outdir}/lime",
                                 class_names=self.class_names, random_samples = self.random_samples,
                                 feature_type = self.feature_type,
                                 global_explanation_sample_num = lime_global_explanation_sample_num,
                                 feature_map_reverse = self.feature_map_reverse,
                                 model_finetune = lime_model_finetune,
                                 fine_tune_cv_nfolds = n_folds,
                                 scoring = scoring,
                                 top_num_features_to_plot = top_num_features_to_plot,
                                 save_results = save_results,
                                 show_plots = show_plots
                                 )
            
        if ("xgb_feature_importance" in self.analyses) or ("randomforest_feature_importance" in self.analyses) or ("xgb_permutation_feature_importance" in self.analyses) or ("rf_permutation_feature_importance" in self.analyses):
            self.FeatureImportance_Analyzer = FeatureImportance_Analysis(
                                              X=self.X,
                                              y=self.y, 
                                              test_size = test_size,
                                              feature_type = feature_type,
                                              top_num_features_to_plot =top_num_features_to_plot, 
                                              feature_map_reverse=self.feature_map_reverse, 
                                              model_finetune = featureimportance_finetune, 
                                              fine_tune_cv_nfolds = n_folds, scoring =scoring, 
                                              outdir = outdir,
                                              save_results = save_results,
                                              show_plots = show_plots)

        if "lasso" in self.analyses:
            self.LassoCoefficient_Analyzer = CoxFeatureRanker(
                                             X=self.X, 
                                             y=self.y, 
                                             C=1.0,
                                             penalty='l1',
                                             l1_ratio=None,
                                             outdir = outdir,
                                             feature_type = feature_type,
                                             feature_map_reverse = self.feature_map_reverse,
                                             model_finetune = cox_model_finetune,
                                             fine_tune_cv_nfolds = n_folds,
                                             scoring = scoring,
                                             top_num_features_to_plot = top_num_features_to_plot,
                                             save_results = save_results,
                                             show_plots = show_plots)
            
        if "ridge" in self.analyses:
            self.RidgeCoefficient_Analyzer = CoxFeatureRanker(
                                             X=self.X, 
                                             y=self.y, 
                                             C=1.0,
                                             penalty='l2',
                                             l1_ratio=None,
                                             outdir = outdir,
                                             feature_type = feature_type,
                                             feature_map_reverse = self.feature_map_reverse,
                                             model_finetune = cox_model_finetune,
                                             fine_tune_cv_nfolds = n_folds,
                                             scoring = scoring,
                                             top_num_features_to_plot = top_num_features_to_plot,
                                             save_results = save_results,
                                             show_plots = show_plots)
            
        if "elasticnet" in self.analyses:
            self.ElasticNetCoefficient_Analyzer = CoxFeatureRanker(
                                                  X=self.X, 
                                                  y=self.y, 
                                                  C=1.0,
                                                  penalty='elasticnet',
                                                  l1_ratio=0.5,
                                                  outdir = outdir,
                                                  feature_type = feature_type,
                                                  feature_map_reverse = self.feature_map_reverse,
                                                  model_finetune = cox_model_finetune,
                                                  fine_tune_cv_nfolds = n_folds,
                                                  scoring = scoring,
                                                  top_num_features_to_plot = top_num_features_to_plot,
                                                  save_results = save_results,
                                                  show_plots = show_plots)
            
        # Initialize top features json object for each analysis
        self.feature_weights = {}

        # Prepare output directories
        directories = [term for term in analyses if 'feature_importance' not in term]
        directories.append("feature_importance")
        for analysis in directories:
            for subdir in ["png", "pdf"]:
                os.makedirs(os.path.join(self.outdir, analysis, subdir), exist_ok=True)


    def perform_shap_analysis(self):
        """
        Executes all SHAP analysis functions from the custom SHAP_Analysis module.
        
        This method sequentially runs the following SHAP analysis functions:
        1. shapWaterFall: Creates waterfall plots for random samples from different classes.
        2. shapForce: Generates force plots for random samples from different classes.
        3. shapSummary: Produces a summary plot of SHAP values.
        4. shapHeatmap: Creates a heatmap of the top differentiating features.
        5. shapFeatureImportance: Computes and returns the top differentiating features based on SHAP values.
        
        The results are saved as PNG and PDF files in the specified output directory.
        """
        try:
            logging.info("Performing SHAP Analysis")
            length = 110
            self.print("="*length)
            self.print(" Starting SHAP Waterfall Analysis ")
            self.print("="*length)
            self.SHAP_Analyzer.shapWaterFall()
            self.fitted_model_xgb = self.SHAP_Analyzer.fitted_model # to make fitted xgb accessible to other analyses for quick computation
            
            self.print("="*length)
            self.print(" Starting SHAP Force Plot Analysis ")
            self.print("="*length)
            self.SHAP_Analyzer.shapForce()
            
            self.print("="*length)
            self.print(" Starting SHAP Summary Plot Analysis ")
            self.print("="*length)
            self.SHAP_Analyzer.shapSummary()
            
            self.print("="*length)
            self.print(" Starting SHAP Heatmap Analysis ")
            self.print("="*length)
            self.SHAP_Analyzer.shapHeatmap()
    
            self.print("="*length)
            self.print(" Starting Mean SHAP Plot Analysis ")
            self.print("="*length)
            self.SHAP_Analyzer.meanSHAP()
            
            self.print("="*length)
            self.print(" Starting SHAP Feature Importance Analysis ")
            self.print("="*length)
            self.feature_weights["shap"] = self.SHAP_Analyzer.shapFeatureImportance()
            self.print(sorted(self.feature_weights["shap"], key=self.feature_weights["shap"].get, reverse=True)[:self.top_num_features_to_plot])
            
            self.print("="*length)
            self.print(" SHAP Analysis Completed ")
            self.print("="*length)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_lime_analysis(self):
        """
        Executes all LIME analysis functions from the custom LIME_Analysis module.
        """
        try:
            logging.info("Performing LIME Analysis")
            length = 110
            self.print("="*length)
            self.print(" Starting LIME Per Sample Explanations")
            self.print("="*length)
            self.LIME_Analyzer.explain_samples()
            self.fitted_model_rf = self.LIME_Analyzer.fitted_model # make fitted random forest available for other analyses for faster computation
            
            self.print("="*length)
            self.print(" Starting LIME Feature Importance Analysis ")
            self.print("="*length)
            self.feature_weights["lime"] = self.LIME_Analyzer.limeFeatureImportance()
            self.print(sorted(self.feature_weights["lime"], key=self.feature_weights["lime"].get, reverse=True)[:self.top_num_features_to_plot])
    
            self.print("="*length)
            self.print(" LIME Analysis Completed ")
            self.print("="*length)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_anova(self):
        """
        Perform ANOVA to identify significantly different features.
        """
        try:
            logging.info("Performing ANOVA Analysis")
            length = 110
            self.print("=" * length)
            self.print(" Starting ANOVA")
            self.print("=" * length)
            
            # Perform ANOVA
            f_statistic, p_values = f_classif(self.X, self.y)
            significant_features = pd.DataFrame({
                "Features": list(map(lambda x: self.feature_map_reverse[x], self.X.columns)), 
                "F-value": f_statistic, 
                "p-value": p_values
            })
            
            # Sort significant features by F-value
            logging.info("Computing ANOVA Features")
            significant_features = significant_features.sort_values(by="F-value", ascending=False).reset_index(drop=True)
            
            # Set colors for the bar plot (all bars blue)
            logging.info("Plotting ANOVA Features")
            top_anova_features = significant_features[significant_features["p-value"] < 0.05].head(20)
            colors = top_anova_features['F-value'].apply(lambda x: "blue")
            
            # Plot the F-values of the top differentiating features
            plt.figure(figsize=(4, 8))
            plt.barh(top_anova_features['Features'], top_anova_features['F-value'], color=colors)
            plt.xlabel('F-value', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f'ANOVA Feature Importance of Top Differentiating {self.feature_type}s', fontsize=15, loc='center', pad=15)
            plt.gca().invert_yaxis()
            
            # Save the plot
            logging.info("Saving Plots") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/anova/png/anova_features_plot.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt.gcf(), f'{self.outdir}/anova/png/anova_features_plot.pdf', 
                       object_type = "figure", data_name = "Figure")
            
            #Show figure
            self.maybe_show(fig=plt)
        
            # Save top features
            self.feature_weights["anova"] = {significant_features.Features[i]: significant_features["F-value"][i] 
                        for i in range(len(significant_features))}
            
            # display the top 25 features
            self.print(sorted(self.feature_weights["anova"], key=self.feature_weights["anova"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" ANOVA Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_t_test(self):
        """
        Perform t-Test to identify significantly different features.
        """
        try:
            logging.info("Performing T-TEST")
            length = 110
            self.print("=" * length)
            self.print(" Starting t-Test")
            self.print("=" * length)
            
            # Sample dataset
            np.random.seed(0)
            
            # Separate the data by class
            class_0 = self.X.iloc[self.labels[self.labels == self.class_names[0]].dropna().index,:]
            class_1 = self.X.iloc[self.labels[self.labels == self.class_names[1]].dropna().index,:]
            
            # Perform t-tests and store p-values
            t_test_values = {"Features": [], "statistic": [], "pvalue": [], "df": []}
            for column in self.X.columns:
                output = ttest_ind(class_0[column], class_1[column])
                t_test_values["Features"].append(column)
                t_test_values["pvalue"].append(output.pvalue)
                t_test_values["statistic"].append(output.statistic)
                t_test_values["df"].append(output.df)
            
            # Convert p-values to a DataFrame for easy sorting and viewing
            logging.info("Computing Feature Importance by Statistical Significance")
            p_values_df = pd.DataFrame(t_test_values)
            p_values_df["Abs(statistic)"] = p_values_df["statistic"].abs()
            p_values_df = p_values_df.sort_values(by = "pvalue", ascending =True)
    
            # reverse map feature names
            p_values_df["Features"] = p_values_df["Features"].apply(lambda x: self.feature_map_reverse[x])
    
            # Set colors for the bar plot (all bars blue)
            logging.info("Plotting Top n Features")
            top_t_test_features = p_values_df[p_values_df.pvalue < 0.05].head(self.top_num_features_to_plot)
            colors = top_t_test_features['pvalue'].apply(lambda x: "blue")
            
            # Plot the F-values of the top differentiating features
            plt.figure(figsize=(4, 8))
            plt.barh(top_t_test_features['Features'], top_t_test_features['pvalue'], color=colors)
            plt.xlabel('P-Value', fontsize=15)
            plt.xticks(fontsize=15, rotation=90)
            plt.yticks(fontsize=15)
            plt.title(f't-Test Feature Importance of Top Differentiating {self.feature_type}s', 
                      fontsize=15, loc='center', pad=15)
            plt.gca().invert_yaxis()
            
            # Save the plot
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f'{self.outdir}/t_test/png/t_test_features_plot.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt.gcf(), f'{self.outdir}/t_test/png/t_test_features_plot.pdf', 
                       object_type = "figure", data_name = "Figure")
            
            # Show figure
            self.maybe_show(fig=plt)
            
            # Save top features
            self.feature_weights["t_test"] = {p_values_df.Features[i]: p_values_df["pvalue"][i] 
                        for i in range(len(p_values_df))}
            
            # Display the top 25 features
            self.print(sorted(self.feature_weights["t_test"], key=self.feature_weights["t_test"].get, reverse=False)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" t-Test Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys) 

    def perform_lasso_analysis(self):
        """
        Perform LASSO Algorithm and extracts lasso coefficients to identify significantly different features.
        """
        try:
            logging.info("Performing LASSO")
            length = 110
            self.print("=" * length)
            self.print(" Starting LASSO Analysis")
            self.print("=" * length)
            
            self.LassoCoefficient_Analyzer.fit()
            # Save top features
            self.feature_weights["lasso"] = self.LassoCoefficient_Analyzer.FeatureImportance()
            
            # Display the top n features
            self.print(sorted(self.feature_weights["lasso"], key=self.feature_weights["lasso"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" LASSO Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_ridge_analysis(self):
        """
        Perform RIDGE Algorithm and extracts coefficients to identify significantly different features.
        """
        try:
            logging.info("Performing RIDGE")
            length = 110
            self.print("=" * length)
            self.print(" Starting RIDGE Analysis")
            self.print("=" * length)
            
            self.RidgeCoefficient_Analyzer.fit()
            # Save top features
            self.feature_weights["ridge"] = self.RidgeCoefficient_Analyzer.FeatureImportance()
            
            # Display the top n features
            self.print(sorted(self.feature_weights["ridge"], key=self.feature_weights["ridge"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" RIDGE Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys) 

    def perform_elasticnet_analysis(self):
        """
        Perform ElasticNet Algorithm and extracts coefficients to identify significantly different features.
        """
        try:
            logging.info("Performing ELASTICNET")
            length = 110
            self.print("=" * length)
            self.print(" Starting ELASTICNET Analysis")
            self.print("=" * length)
            
            self.ElasticNetCoefficient_Analyzer.fit()
            # Save top features
            self.feature_weights["elasticnet"] = self.ElasticNetCoefficient_Analyzer.FeatureImportance()
            
            # Display the top n features
            self.print(sorted(self.feature_weights["elasticnet"], key=self.feature_weights["elasticnet"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" ELASTICNET Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_randomforest_feature_importance_analysis(self):
        """
        Executes feature importance analysis using a Random Forest classifier and self.prints the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        try:
            self.FeatureImportance_Analyzer.fitted_model_rf = self.fitted_model_rf # fetch for already fitted model from prevous analyses if any.
            
            logging.info("Performing Random Forest Feature Importance Analysis")
            
            length = 110
            self.print("=" * length)
            self.print(" Starting Feature Importance Analysis with Random Forest")
            self.print("=" * length)
    
            # 
            self.feature_weights["randomforest_feature_importance"] = self.FeatureImportance_Analyzer.RandomForestFeatureImportance()
            
            # Display the top n features
            self.print(sorted(self.feature_weights["randomforest_feature_importance"], key=self.feature_weights["randomforest_feature_importance"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" Random Forest Feature Importance Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_xgb_feature_importance_analysis(self):
        """
        Executes feature importance analysis using an XGBoost classifier and displays the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        try:
            self.FeatureImportance_Analyzer.fitted_model_xgb = self.fitted_model_xgb # fetch for already fitted model from prevous analyses if any.
            logging.info("Performing XGBoost Feature Importance Analysis")
            length = 110
            self.print("=" * length)
            self.print(" Starting Feature Importance Analysis with XGBoost")
            self.print("=" * length)
    
            # 
            self.feature_weights["xgb_feature_importance"] = self.FeatureImportance_Analyzer.XGBoostFeatureImportance()
            
            # Display the top 25 features
            self.print(sorted(self.feature_weights["xgb_feature_importance"], key=self.feature_weights["xgb_feature_importance"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(" XGBoost Feature Importance Analysis Completed ")
            self.print("=" * length)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_permutation_feature_importance_analysis(self, model_name = "Random Forest"):
        """
        Executes feature importance analysis using a Random Forest classifier, performs permutation feature importance analysis and displays the top N features based on their importance scores. It logs the process and displays a formatted message indicating the start and completion of the analysis.
        """
        try:
            # fetch for already fitted models from prevous analyses if any.
            self.FeatureImportance_Analyzer.fitted_model_rf = self.fitted_model_rf 
            self.FeatureImportance_Analyzer.fitted_model_xgb = self.fitted_model_xgb 
            logging.info(f"Performing Permutation Feature Importance Analysis with {model_name}")
            length = 110
            self.print("=" * length)
            self.print(f" Starting Feature Importance Analysis with Permutation Method with {model_name}")
            self.print("=" * length)
    
            # 
            mt = "rf" if model_name == "Random Forest" else "xgb"
            self.feature_weights[f"{mt}_permutation_feature_importance"] = self.FeatureImportance_Analyzer.PermutationFeatureImportance(model_name = model_name)
            
            # Display the top 25 features
            self.print(sorted(self.feature_weights[f"{mt}_permutation_feature_importance"], key=self.feature_weights[f"{mt}_permutation_feature_importance"].get, reverse=True)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(f" Permutation Feature Importance Analysis wıth {model_name} Completed ")
            self.print("=" * length)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_limma_analysis(self):
        """
        Perform differential expression analysis using the LIMMA method.
    
        This method:
        - Initializes the LIMMA wrapper with the configured data and parameters.
        - Executes the R-based LIMMA pipeline.
        - Retrieves adjusted p-values (feature weights) for each feature.
        - Stores the result in `self.feature_weights["limma"]`.
        - Prints the top N features ranked by significance (lowest adjusted p-values).
    
        Raises:
            CustomException: If the LIMMA pipeline fails or input data is invalid.
        """

        try:
            logging.info(f"Performing Limma Analysis")
            length = 110
            self.print("=" * length)
            self.print(f" Performing Limma Analysis")
            self.print("=" * length)
    
            # Initialize and run LIMMA analysis
            limma = LIMMA(data=self.data,
                          labels_column=self.labels_column,
                          reference_class=self.reference_class,
                          sample_id_column=self.sample_id_column,
                          feature_type=self.feature_type,
                          r_script_path="./RScripts/run_limma.R")

            # Perform analysis and store adjusted p-values
            _, limma_feature_weights_= limma.fit()
            self.feature_weights["limma"] = limma_feature_weights_
            
            # Display top N features based on lowest adjusted p-values
            self.print(sorted(self.feature_weights["limma"], key=self.feature_weights["limma"].get, reverse=False)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(f" Limma Analysis Completed ")
            self.print("=" * length)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
            
    def perform_boruta_analysis(self):
        """
        Perform feature selection using the Boruta algorithm.
    
        This method:
        - Initializes the BorutaSelector with the configured data (excluding sample ID column).
        - Runs the Boruta feature selection algorithm using a RandomForestClassifier backend.
        - Stores the feature ranking (lower rank = more important) in `self.feature_weights["boruta"]`.
        - Prints the top N features with the lowest ranks.
    
        Raises:
            CustomException: If Boruta fails due to data errors or execution issues.
        """

        try:
            logging.info(f"Performing Boruta Analysis")
            length = 110
            self.print("=" * length)
            self.print(f" Performing Boruta Analysis")
            self.print("=" * length)
    
            # Initialize and run Boruta analysis
            selector = BorutaSelector(
                data=self.data.drop(columns = self.sample_id_column),
                labels_column=self.labels_column
            )

             # Perform analysis and store ranks
            rank_map, _ = selector.fit()
            self.feature_weights["boruta"] = rank_map
            
            # Display top N features based on lowest Rank
            self.print(sorted(self.feature_weights["boruta"], key=self.feature_weights["boruta"].get, reverse=False)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(f" Boruta Analysis Completed ")
            self.print("=" * length)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_mannwhitneyu_analysis(self):
        """
        Perform non-parametric univariate feature scoring using the Mann–Whitney U test.
    
        This method:
        - Initializes the MannWhitneySelector with the configured data.
        - Performs a two-sided Mann–Whitney U test on each feature to compare distributions between two groups.
        - Stores the resulting p-values (feature significance scores) in `self.feature_weights["mannwhitneyu"]`.
        - Prints the top N features with the lowest p-values (most significant differences).
    
        Raises:
            CustomException: If the test fails due to invalid data or runtime errors.
        """
        try:
            logging.info(f"Performing Mann–Whitney U test")
            length = 110
            self.print("=" * length)
            self.print(f" Performing Mann–Whitney U test")
            self.print("=" * length)
    
            # Initialize and run Mann–Whitney U test
            selector = MannWhitneySelector(
                data=self.data.drop(columns = self.sample_id_column),
                labels_column=self.labels_column
            )
            pval_map = selector.fit()
            self.feature_weights["mannwhitneyu"] = pval_map
            
            # Display top N features based on lowest pvalue
            self.print(sorted(self.feature_weights["mannwhitneyu"], key=self.feature_weights["mannwhitneyu"].get, reverse=False)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(f" Mann–Whitney U test Completed ")
            self.print("=" * length)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def perform_svm_rfe_analysis(self):
        """
        Perform feature selection using SVM-RFE (Support Vector Machine with Recursive Feature Elimination).
    
        This method:
        - Initializes the SVMRFESelector with the provided dataset and parameters.
        - Runs recursive feature elimination using a linear SVM classifier.
        - Stores the resulting feature ranking (1 = most important) in `self.feature_weights["svm_rfe"]`.
        - Prints the top N features based on the lowest ranks.
    
        Raises:
            CustomException: If the SVM-RFE process fails due to invalid input or model error.
        """
        try:
            logging.info(f"Performing SVM-RFE")
            length = 110
            self.print("=" * length)
            self.print(f" Performing SVM-RFE")
            self.print("=" * length)
    
            # Initialize and run SVM-RFE
            selector = SVMRFESelector(
                data=self.data.drop(columns = self.sample_id_column),
                labels_column=self.labels_column,
                n_features_to_select=self.top_num_features_to_plot 
            )
            
            rank_map, _ = selector.fit()
            self.feature_weights["svm_rfe"] = rank_map
            
            # Display top N features based on lowest pvalue
            self.print(sorted(self.feature_weights["svm_rfe"], key=self.feature_weights["svm_rfe"].get, reverse=False)[:self.top_num_features_to_plot])
            self.print("=" * length)
            self.print(f" SVM-RFE Analysis Completed ")
            self.print("=" * length)
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
            
    def run_all_analyses(self):
        """
        Executes all analyses specified in the 'analyses' attribute of the class.
        This method sequentially runs SHAP, LIME, ANOVA, t-test, and Feature Importance analyses.
        
        The results from each analysis are saved in the designated output directories, 
        and the top differentiating features are stored in the 'feature_weights' attribute.
        """

        try:
            logging.info("RUNNING ALL ANALYSES")
            length = 110
            self.print("="*length)
            self.print(" Starting All Analyses ")
            self.print("="*length)
            
            if "shap" in self.analyses:
                self.perform_shap_analysis()
            
            if "lime" in self.analyses:
                self.perform_lime_analysis()
            
            if "anova" in self.analyses:
                self.perform_anova()
            
            if "t_test" in self.analyses:
                self.perform_t_test()
            
            if "randomforest_feature_importance" in self.analyses:
                self.perform_randomforest_feature_importance_analysis()
    
            if "xgb_feature_importance" in self.analyses:
                self.perform_xgb_feature_importance_analysis()
    
            if "rf_permutation_feature_importance" in self.analyses:
                self.perform_permutation_feature_importance_analysis()

            if "xgb_permutation_feature_importance" in self.analyses:
                self.perform_permutation_feature_importance_analysis(model_name = "XGBClassifier")
                
            if "lasso" in self.analyses:
                self.perform_lasso_analysis() 

            if "ridge" in self.analyses:
                self.perform_ridge_analysis()

            if "elasticnet" in self.analyses:
                self.perform_elasticnet_analysis()
            if "limma" in self.analyses:
                self.perform_limma_analysis()
            if "boruta" in self.analyses:
                self.perform_boruta_analysis()
            if "mannwhitneyu" in self.analyses:
                self.perform_mannwhitneyu_analysis()
            if "svm_rfe" in self.analyses:
                self.perform_svm_rfe_analysis()
    
            # save self.feature_weights convert all values to floats
            for a in self.feature_weights.keys():
                for feature in self.feature_weights[a].keys():
                    self.feature_weights[a][feature] = float(self.feature_weights[a][feature])
    
            logging.info("Saving Feature Importances") 
            self.maybe_save(self.feature_weights, f"{self.outdir}/feature_importances.json", 
                            object_type = "json", data_name = "Feature Weights") 
            
            self.print("="*length)
            self.print(" All Analyses Completed ")
            self.print("="*length)
    
            logging.info("ALL ANALYSES COMPLETED!!!")
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)