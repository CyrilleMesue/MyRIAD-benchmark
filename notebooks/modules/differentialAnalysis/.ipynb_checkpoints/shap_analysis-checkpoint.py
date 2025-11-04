# Load general packages
import pandas as pd
import numpy as np
import os
import sys
from tqdm.notebook import tqdm

# Load sklearn packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import set_config

# Load visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Load specialized packages
import shap

# Load Custom Modules
from modules.utils import save_json, load_json, getparams
from modules.exception import CustomException
from modules.logger import logging
from modules.verbose_config import BaseConfigurable

# Set sklearn configuration
set_config(transform_output="pandas")  # Set SHAP output format to pandas DataFrame for easier manipulation


class SHAP_Analysis(BaseConfigurable):
    """
    This class is responsible for performing SHAP analysis and creating various visualization plots.

    Attributes:
    -----------
    X : pd.DataFrame
        The feature matrix used for model training and SHAP analysis.
    y : pd.Series or np.array
        The target labels corresponding to the feature matrix X.
    outdir : str
        The directory where output plots and files will be saved.
    random_samples : dict
        A dictionary containing random sample indices for different classes.
    biomarker_type : str
        The type of biomarker being analyzed (e.g., microRNA, gene, metabolite).
    shap_values : np.array
        The SHAP values computed for the feature matrix X.
    fitted_model : XGBClassifier
        The trained model used to generate SHAP values.
    scoring : str
        The scoring metric used for model evaluation. Options include "f1", "recall", "precision", "accuracy".
    feature_map_reverse : dict
        A mapping of feature names to their original names.
    model_finetune : bool
        Indicates whether to fine-tune the model.
    fine_tune_cv_nfolds : int
        The number of folds to use for cross-validation during model fine-tuning.
    top_num_features_to_plot : int 
        The number of features to display on plots

    Methods:
    --------
    shapWaterFall():
        Plots a waterfall plot for both control and disease samples side by side.
    
    shapForce():
        Plots force plots for two random samples from different classes.
    
    shapSummary():
        Creates and saves a SHAP summary plot.
    
    shapHeatmap():
        Creates and saves a SHAP heatmap plot.
    
    shapFeatureImportance():
        Computes and returns the top differentiating features based on SHAP values.
    """

    def __init__(self, 
                 X=None, 
                 y=None, 
                 outdir: str = "shap", 
                 random_samples: dict = None,
                 feature_type: str = "microRNA",
                 feature_map_reverse:dict = None,
                 model_finetune:bool = False,
                 fine_tune_cv_nfolds:int = 5,
                 scoring:str = "f1",
                 top_num_features_to_plot:int = 20,
                 save_results:bool = True,
                 show_plots:bool = True
                ):

        """
        Initializes the SHAP_Analysis object with the provided parameters.

        Parameters:
        -----------
        X : pd.DataFrame, optional
            The feature matrix used for model training and SHAP analysis.
        y : pd.Series or np.array, optional
            The target labels corresponding to the feature matrix X.
        outdir : str, optional
            The directory where output plots and files will be saved. Default is "shap".
        random_samples : dict, optional
            A dictionary containing random sample indices for different classes.
        feature_type : str, optional
            The type of feature being analyzed (e.g., microRNA, gene, metabolite). Default is "microRNA".
        feature_map_reverse : dict, optional
            A mapping of feature names to their original names.
        model_finetune : bool, optional
            Indicates whether to fine-tune the model. Default is False.
        fine_tune_cv_nfolds : int, optional
            The number of folds to use for cross-validation during model fine-tuning. Default is 5.
        scoring : str, optional
            The scoring metric used for model evaluation. Options include "f1", "recall", "precision", "accuracy". Default is "f1".
        top_num_features_to_plot : int, optional
            The number of features to display per plot.
        """
        self.X = X
        self.y = y
        self.random_samples = random_samples
        self.outdir = outdir
        self.feature_type = feature_type
        self.feature_map_reverse = feature_map_reverse
        self.model_finetune =  model_finetune
        self.fine_tune_cv_nfolds = fine_tune_cv_nfolds
        self.scoring = scoring
        self.top_num_features_to_plot = top_num_features_to_plot
        self.fitted_model = None
        self.save_results = save_results
        self.show_plots = show_plots

    
    def fit(self):
        """
        Fits the XGBoost classifier to the training data and explains the model predictions using SHAP.
    
        This method performs the following steps:
        1. Fits an XGBoost classifier to the training data (`self.X`, `self.y`).
        2. Uses SHAP (SHapley Additive exPlanations) to explain the model predictions.
        3. Maps the SHAP feature names to their original names using `feature_map_reverse`.
    
        Attributes:
        self.fitted_model : XGBClassifier
            The fitted XGBoost classifier model.
        self.shap_values : shap.Explanation
            The SHAP values for the training data, with feature names mapped to their original names.
        """

        try:
    
            if self.model_finetune: # Check if model fine-tuning is enabled
                model = XGBClassifier() # initialize model
                params = getparams()["XGBClassifier"]
                kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, random_state=42, shuffle = True)
    
                logging.info("Fine tuning XGBClassifier") # Log the start of model fine-tuning process
                gs = GridSearchCV(model,params,cv=kfold, scoring = self.scoring)
                gs.fit(self.X,self.y) #  fit grid search object for parameter tunning
                model_best_params = gs.best_params_
                model.set_params(**model_best_params) # Update model with optimal parameters
                self.fitted_model = model.fit(self.X, self.y)
            else:
                self.fitted_model = XGBClassifier().fit(self.X, self.y)
                
            # Explain the model predictions using SHAP
            explainer = shap.Explainer(self.fitted_model)
            self.shap_values = explainer(self.X) # get shap explanations
            self.shap_values.feature_names = list(map(lambda x:self.feature_map_reverse[x], 
                                                      self.shap_values.feature_names)) # map features to their original names [modified names are provided to avoid any pandas issues]

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
        
    def _check_fit(self):
        """
        Checks if the model is fitted and the SHAP explainer is initialized.
        If not, it calls the fit method to train the model.
        """
        try:
            if not hasattr(self, 'shap_values') or self.shap_values is None: # Ensure SHAP values are computed before plotting
                logging.info("Training XGBClassifier for SHAP Analysis")
                self.fit()
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def shapWaterFall(self):
        """
        Plot waterfall plots for a control and disease sample side by side.
        The plots are saved as PNG and PDF files in the specified output directory.
        """
        try:
            self._check_fit()
            logging.info("Plotting Waterfall Plots") # Log that the waterfall plots are being created
            fig, axes = plt.subplots(1, 2, figsize=(30, 12))
    
            # Waterfall plot for class 0 sample
            class_names = list(self.random_samples.keys())
            plt.sca(axes[0])  # Set the current axis to the first subplot
            shap.plots.waterfall(self.shap_values[self.random_samples[class_names[0]]], max_display=self.top_num_features_to_plot, show=False)
            axes[0].set_title(f"{class_names[0]} Sample")
    
            # Waterfall plot for a class 1 sample
            plt.sca(axes[1])  # Set the current axis to the second subplot
            shap.plots.waterfall(self.shap_values[self.random_samples[class_names[1]]], max_display=self.top_num_features_to_plot, show=False)
            axes[1].set_title(f"{class_names[1]} Sample")
    
            plt.subplots_adjust(wspace=2.5)  # Adjust horizontal space between subplots
            plt.tight_layout()
    
            logging.info("Saving Plots")
            # save plots
            self.maybe_save(plt.gcf(), f"{self.outdir}/png/shap_waterfall_subplots.png", 
                       object_type = "figure", data_name = "Figure") # Save waterfall plot as PNG
            self.maybe_save(plt.gcf(), f"{self.outdir}/pdf/shap_waterfall_subplots.pdf", 
                       object_type = "figure", data_name = "Figure") # Save waterfall plot as PDF
            
            # Show figure
            self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def shapForce(self):
        """
        Plot force plots for two random samples from different classes.
        The plots are saved as PNG and PDF files in the specified output directory.
        """

        try:
            self._check_fit()
            logging.info("Plotting Force Plots")
            class_names = list(self.random_samples.keys())
    
            # Plot for class 0
            shap.plots.force(self.shap_values[self.random_samples[class_names[0]]], matplotlib=True, show=False)
            plt.title(f"Force Plot for 1 {class_names[0]} Sample", y=1.5)
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f"{self.outdir}/png/forceplot_for_{class_names[0]}_sample.png", 
                       object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f"{self.outdir}/png/forceplot_for_{class_names[0]}_sample.pdf", 
                       object_type = "figure", data_name = "Figure")
    
            # Plot for class 1
            shap.plots.force(self.shap_values[self.random_samples[class_names[1]]], matplotlib=True, show=False)
            plt.title(f"Force Plot for 1 {class_names[1]} Sample", y=1.5)
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f"{self.outdir}/png/forceplot_for_{class_names[1]}_sample.png", 
                       object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f"{self.outdir}/png/forceplot_for_{class_names[1]}_sample.pdf", 
                       object_type = "figure", data_name = "Figure")
    
            # Show figure
            self.maybe_show(fig=plt)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def shapSummary(self):
        """
        Create and save a SHAP summary plot.
        The plot is saved as both PNG and PDF files in the specified output directory.
        """

        try:
            self._check_fit()
            logging.info("Plotting SHAP Summary Plot")
            plt.figure(figsize=(8, 6), dpi=300)
            shap.summary_plot(self.shap_values, self.X, show=False)
            plt.title(f"SHAP Summary Plot of Top Differentiating {self.feature_type}s", fontsize=20, loc='center', pad=20)
    
            plt.xlabel('SHAP value (impact on model output)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            
            # Get the current figure
            fig = plt.gcf()
            
            # Modify the color bar
            colorbar = fig.axes[-1]  # Access the color bar (typically the last axis)
            colorbar.tick_params(labelsize=20)  # Set the font size of the ticks
            colorbar.set_ylabel('Feature value', fontsize=20)  # Set the font size of the label
    
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/shap_summary_plot.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/shap_summary_plot.pdf', 
                       object_type = "figure", data_name = "Figure")
            
            # Show figure
            self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def shapHeatmap(self):
        """
        Create and save a SHAP heatmap plot.
        The plot is saved as both PNG and PDF files in the specified output directory.
        """

        try:
            self._check_fit()
            logging.info("Plotting SHAP Heatmap")
            shap.plots.heatmap(self.shap_values, max_display=self.top_num_features_to_plot, show=False)
            plt.title(f"SHAP Heatmap of Top Differentiating {self.feature_type}s", fontsize=15, loc='center', pad=20)
    
            plt.xlabel('Instances', fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            
            # Get the current figure
            fig = plt.gcf()
            
            # Modify the color bar
            colorbar = fig.axes[-1]  # Access the color bar (typically the last axis)
            colorbar.tick_params(labelsize=15)  # Set the font size of the ticks
            colorbar.set_ylabel('SHAP value (impact on model output)', fontsize=15)  # Set the font size of the label
    
            logging.info("Saving Plots") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/shap_heatmap_plot.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/shap_heatmap_plot.pdf', 
                       object_type = "figure", data_name = "Figure")
            
            # Show figure
            self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def meanSHAP(self):
        """
        Create and save a mean SHAP plot.
        The plot is saved as both PNG and PDF files in the specified output directory.
        """

        try:
            self._check_fit()
            logging.info("Plotting Mean SHAP Plot")
            plt.figure(figsize=(8, 6), dpi=300)
            shap.plots.bar(self.shap_values, max_display=self.top_num_features_to_plot, show = False)
            plt.title(f"Mean SHAP Plot of Top Differentiating {self.feature_type}s", fontsize=15, loc='center', pad=20)
    
            plt.xlabel('mean(|SHAP value|)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
    
            logging.info("Saving Plots") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/mean_shap_plot.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/mean_shap_plot.pdf', 
                       object_type = "figure", data_name = "Figure")
            
            # Show figure
            self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def shapFeatureImportance(self):
        """
        Compute and return the top differentiating features based on SHAP values.
        Returns a dictionary with the top 25, 50, 100, and all features ordered by importance.
        """

        try:
            self._check_fit()
            logging.info("Computing SHAP Feature Importances")
            shap_values_df = pd.DataFrame(self.shap_values.values, columns=list(map(lambda x:self.feature_map_reverse[x],self.X.columns))) # Create DataFrame for SHAP values
            shap_importance = shap_values_df.abs().mean()
    
            top_features = {shap_importance.index[i]: float(shap_importance[i]) for i in range(len(shap_importance))}

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
        return top_features