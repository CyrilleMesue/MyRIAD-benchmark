# Load general packages
import pandas as pd
import numpy as np
import os
import sys
from tqdm.notebook import tqdm
import contextlib

# Load sklearn packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import set_config

# Load visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Load Custom Modules
from modules.utils import save_json, load_json, getparams
from modules.logger import logging
from modules.exception import CustomException
from modules.verbose_config import BaseConfigurable


class FeatureImportance_Analysis(BaseConfigurable):

    """
    A class for analyzing feature importance using various classification models.

    This class provides methods to compute and visualize feature importance scores 
    from trained classifiers such as XGBoost and Random Forest. It supports model 
    fine-tuning through cross-validation and allows for plotting of the top features.

    Attributes:
        X (pd.DataFrame): The feature matrix containing input data for the model.
        y (pd.Series): The target variable corresponding to the feature matrix.
        feature_map_reverse (dict): A mapping from encoded feature names to their original names.
        feature_type (str): A string representing the type of features (e.g., 'gene', 'protein').
        top_num_features_to_plot (int): The number of top features to visualize in the plots.
        model_finetune (bool): Flag indicating whether to perform model fine-tuning.
        fine_tune_cv_nfolds (int): The number of folds for cross-validation during fine-tuning.
        scoring (str): The metric to optimize during model training and evaluation.
        outdir (str): The output directory for saving plots and results.
        X_train (pd.DataFrame): The training feature matrix after train-test split.
        X_test (pd.DataFrame): The testing feature matrix after train-test split.
        y_train (pd.Series): The training target variable after train-test split.
        y_test (pd.Series): The testing target variable after train-test split.

    Methods:
        PermutationFeatureImportance:
            Fits a RandomForest classifier and computes permutation feature importance 
            on the test dataset. If model fine-tuning is enabled, performs cross-validation 
            to find the optimal hyperparameters before fitting the model. Otherwise, fits 
            the default RandomForest model. The method calculates permutation feature importances, 
            visualizes the results with box plots, and saves the plots as PNG and PDF files.
            Returns a dictionary mapping feature names to their importance scores, sorted 
            in descending order.

        RandomForestFeatureImportance:
            Fits a RandomForest classifier to the training data and plots the top N differentiating 
            features. If model fine-tuning is enabled, performs cross-validation to find the optimal 
            hyperparameters before fitting the model. Otherwise, fits the default RandomForest model 
            with pre-set parameters. The top N features are plotted and saved as PNG and PDF files.
            Returns a dictionary mapping feature names to their importance scores, sorted in descending order.

        XGBoostFeatureImportance:
            Fits an XGBoost classifier on the training data and plots the top differentiating features. 
            If model fine-tuning is enabled, performs cross-validation to find the optimal hyperparameters 
            before fitting the model. Otherwise, fits the default XGBoost model. The top N features are 
            plotted and saved as PNG and PDF files. The feature importances are computed using the model's 
            fitted coefficients. Returns a dictionary mapping feature names to their importance scores, 
            sorted in descending order.

    Parameters:
        X (pd.DataFrame, optional): The input feature data. Defaults to None.
        y (pd.Series, optional): The target labels corresponding to the input data. Defaults to None.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        feature_map_reverse (dict, optional): A mapping for reversing encoded feature names. Defaults to None.
        feature_type (str, optional): A description of the feature type for reporting purposes. Defaults to None.
        top_num_features_to_plot (int, optional): Number of top features to visualize. Defaults to 20.
        model_finetune (bool, optional): Enable model fine-tuning. Defaults to False.
        fine_tune_cv_nfolds (int, optional): Number of cross-validation folds for fine-tuning. Defaults to 5.
        scoring (str, optional): Scoring metric for model evaluation. Defaults to "f1".
        outdir (str, optional): Directory to save output plots and results. Defaults to "output".

    Example:
        >>> feature_analysis = FeatureImportance_Analysis(X=data_features, y=data_labels)
        >>> feature_analysis.RandomForestFeatureImportance()
    """

    def __init__(self, 
                 X=None,
                 y=None, 
                 test_size:float = 0.2, 
                 feature_map_reverse:dict = None, 
                 feature_type:str = None, 
                 top_num_features_to_plot:int = 20, 
                 model_finetune:bool = False, 
                 fine_tune_cv_nfolds:int = 5,
                 scoring:str = "f1",
                 outdir:str = "output",
                 save_results:bool = True,
                 show_plots:bool = True
                ):
        """
        Initializes the FeatureImportance_Analysis class with the given parameters.

        Parameters:
            X (pd.DataFrame, optional): The input feature data. Defaults to None.
            y (pd.Series, optional): The target labels corresponding to the input data. Defaults to None.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            feature_map_reverse (dict, optional): A mapping for reversing encoded feature names. Defaults to None.
            feature_type (str, optional): A description of the feature type for reporting purposes. Defaults to None.
            top_num_features_to_plot (int, optional): Number of top features to visualize. Defaults to 20.
            model_finetune (bool, optional): Enable model fine-tuning. Defaults to False.
            fine_tune_cv_nfolds (int, optional): Number of cross-validation folds for fine-tuning. Defaults to 5.
            scoring (str, optional): Scoring metric for model evaluation. Defaults to "f1".
            outdir (str, optional): Directory to save output plots and results. Defaults to "output".

        Initializes the training and testing sets based on the provided input data.
        """

        self.X = X
        self.y = y
        self.feature_map_reverse = feature_map_reverse
        self.feature_type = feature_type
        self.top_num_features_to_plot = top_num_features_to_plot
        self.model_finetune = model_finetune 
        self.fine_tune_cv_nfolds = fine_tune_cv_nfolds
        self.scoring = scoring
        self.outdir = outdir
        self.fitted_model_xgb = None
        self.fitted_model_rf = None 
        self.save_results = save_results
        self.show_plots = show_plots

        # create train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, test_size=test_size, random_state=42, shuffle= True)


    def XGBoostFeatureImportance(self):
        """
        Fits an XGBoost classifier on the training data and plots the top differentiating features.
        
        If model fine-tuning is enabled, performs cross-validation to find the optimal hyperparameters
        before fitting the model. Otherwise, fits the default XGBoost model. The top N features are 
        plotted and saved as PNG and PDF files. The feature importances are computed using the model's 
        fitted coefficients.
    
        Returns:
            dict: A dictionary mapping feature names to their importance scores, sorted in descending 
            order.
        """

        try:
            if self.fitted_model_xgb is None:
                logging.info("Fitting XGBClassifier")
                if self.model_finetune: # Check if model fine-tuning is enabled
                    model = XGBClassifier() # initialize model
                    params = getparams()["XGBClassifier"]
                    kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, random_state=42, shuffle = True)
        
                    logging.info("Fine tuning XGBClassifier") # Log the start of model fine-tuning process
                    gs = GridSearchCV(model,params,cv=kfold, scoring = self.scoring)
                    gs.fit(self.X_train,self.y_train) #  fit grid search object for parameter tunning
                    model_best_params = gs.best_params_
                    model.set_params(**model_best_params) # Update model with optimal parameters
                    self.fitted_model_xgb = model.fit(self.X_train,self.y_train)
                else:
                    self.fitted_model_xgb = XGBClassifier(random_state = 42).fit(self.X_train, self.y_train)

            else:
                logging.info("Using Prefitted XGBClassifier")
                
            # plot and save top n features
            logging.info("Computing Feature Importances")
            feature_importance_xgb = pd.Series(self.fitted_model_xgb.feature_importances_, 
                                            index = list(map(lambda x: self.feature_map_reverse[x],self.X_train.columns))).sort_values(ascending = False)
            feature_importance_xgb_n = feature_importance_xgb.head(self.top_num_features_to_plot)
            colors = feature_importance_xgb_n.apply(lambda x: "blue")
            
            # Plot the Feature Importance values of the top differentiating features
            logging.info("Plotting Feature Importances")
            plt.figure(figsize=(4, 8))
            plt.barh(feature_importance_xgb_n.index, feature_importance_xgb_n, color=colors)
            plt.xlabel('Feature Importances',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f'XGBoost Feature Importance of Top Differentiating {self.feature_type}s',
                     fontsize=15, loc='center', pad=15)
            plt.gca().invert_yaxis()
            
            # save
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/xgb_features_plot.png', 
                            object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/xgb_features_plot.pdf', 
                            object_type = "figure", data_name = "Figure") 
            
            # Show figure
            self.maybe_show(fig=plt)
    
            # Save top features
            top_features = {feature_importance_xgb.index[i]:feature_importance_xgb[i] 
                        for i in range(len(feature_importance_xgb))}

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return top_features
    
    def RandomForestFeatureImportance(self, model = None):
        """
        Fits a RandomForest classifier to the training data and plots the top N differentiating features.
    
        If model fine-tuning is enabled, performs cross-validation to find the optimal hyperparameters 
        before fitting the model. Otherwise, fits the default RandomForest model with pre-set parameters. 
        The top N features are plotted and saved as PNG and PDF files.
    
        Returns:
            dict: A dictionary mapping feature names to their importance scores, sorted in descending order.
        """

        try:

            if self.fitted_model_rf is None:
                logging.info("Fitting RandomForestClassifier")
                if self.model_finetune: # Check if model fine-tuning is enabled
                    model = RandomForestClassifier() # initialize model
                    params = getparams()["Random Forest"]
                    kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, random_state=42, shuffle = True)
        
                    logging.info("Fine tuning RandomForestClassifier") # Log the start of model fine-tuning process
                    gs = GridSearchCV(model,params,cv=kfold, scoring = self.scoring)
                    gs.fit(self.X_train,self.y_train) #  fit grid search object for parameter tunning
                    model_best_params = gs.best_params_
                    model.set_params(**model_best_params) # Update model with optimal parameters
                    self.fitted_model_rf = model.fit(self.X_train,self.y_train)
                else:
                    self.fitted_model_rf = RandomForestClassifier(n_estimators=100,min_samples_leaf = 1,random_state = 42).fit(self.X_train, self.y_train)

            else:
                logging.info("Using Prefitted RandomForestClassifier")
                
            # plot and save top n features
            logging.info("Computing Feature Importances")
            feat_importances_rf = pd.Series(self.fitted_model_rf.feature_importances_, 
                                            index = list(map(lambda x: self.feature_map_reverse[x], self.X_train.columns))).sort_values(ascending = False)
            feat_importances_rf_n = feat_importances_rf.head(self.top_num_features_to_plot)
            colors = feat_importances_rf_n.apply(lambda x: "blue")
            
            # Plot the Feature Importance values of the top differentiating features
            logging.info("Plotting Feature Importances")
            plt.figure(figsize=(4, 8))
            plt.barh(feat_importances_rf_n.index, feat_importances_rf_n, color=colors)
            plt.xlabel('Feature Importances',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f'RandomForest Feature Importance of Top Differentiating {self.feature_type}s', 
                      fontsize=15, loc='center', pad=15)
            plt.gca().invert_yaxis()
            
            # save
            logging.info("Saving Plots")
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/rf_features_plot.png', 
                            object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/rf_features_plot.pdf', 
                            object_type = "figure", data_name = "Figure") 
            
            # Show figure
            self.maybe_show(fig=plt)
    
            # Save top features
            top_features = {feat_importances_rf.index[i]:feat_importances_rf[i] 
                        for i in range(len(feat_importances_rf))}
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return top_features

        
    def PermutationFeatureImportance(self, model_name = "Random Forest"):
        """
        Fits a RandomForest classifier and computes permutation feature importance on the test dataset.
    
        If model fine-tuning is enabled, performs cross-validation to find the optimal hyperparameters 
        before fitting the model. Otherwise, fits the default RandomForest model. The method calculates 
        permutation feature importances, visualizes the results with box plots, and saves the plots as 
        PNG and PDF files.
    
        Returns:
            dict: A dictionary mapping feature names to their importance scores, sorted in descending order.
        """

        try:
            models = {"Random Forest": RandomForestClassifier(random_state = 42), 
                      "XGBClassifier": XGBClassifier(random_state = 42)}
            
            if model_name == "Random Forest":
                self.fitted_model = self.fitted_model_rf
            elif model_name == "XGBClassifier":
                self.fitted_model = self.fitted_model_xgb
                
            if self.model_finetune: # Check if model fine-tuning is enabled
                if self.fitted_model is None:
                    logging.info(f"Fitting {model_name}")
                    
                    model = models[model_name] # initialize model
                    params = getparams()[model_name]
                    kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, random_state=42, shuffle = True)
        
                    logging.info(f"Fine tuning {model_name}") # Log the start of model fine-tuning process
                    gs = GridSearchCV(model,params,cv=kfold, scoring = self.scoring)
                    gs.fit(self.X_train,self.y_train) #  fit grid search object for parameter tunning
                    model_best_params = gs.best_params_
                    model.set_params(**model_best_params) # Update model with optimal parameters
                    self.fitted_model_rf = model.fit(self.X_train,self.y_train)

                else:
                    logging.info(f"Using Prefitted {model_name}")
            else:
                self.fitted_model = models[model_name].fit(self.X_train, self.y_train)
    
            # calculate permutation importance for test data 
            logging.info("Running Permutation Importance Algorithm")
            result_test = permutation_importance(
                self.fitted_model, self.X_test, self.y_test, n_repeats=20, random_state=42, n_jobs=8
            )
            
            sorted_importances_idx_test = result_test.importances_mean.argsort()
            importances_test = pd.DataFrame(
                result_test.importances[sorted_importances_idx_test].T,
                columns=self.X.columns[sorted_importances_idx_test],
            )
            importances_test.columns = list(map(lambda x: self.feature_map_reverse[x], importances_test.columns)) # reverse map features to get original names
    
            feat_importances_permutation = importances_test.sum(axis=0).sort_values(ascending=False)
            important_features_by_permutation = feat_importances_permutation.head(self.top_num_features_to_plot).index
            
            # Plot the Feature Importance values of the top differentiating features
            logging.info("Plotting Feature Importances")

            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                plt.figure(figsize=(4, 8))
                
                importances_test[important_features_by_permutation[::-1]].plot.box(vert=False, whis=10)
                plt.title(f'Permutation Feature Importance of Top Differentiating {self.feature_type}s by {model_name}',
                         fontsize=20, loc='center', pad=20)
                plt.axvline(x=0, color="k", linestyle="--")
                plt.xlabel("Decrease in accuracy score",fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
            
            # save
            logging.info("Saving Plots") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/permutation_with_{model_name}_features_plot.png', 
                            object_type = "figure", data_name = "Figure") 
            
            self.maybe_save(plt.gcf(), f'{self.outdir}/feature_importance/png/permutation_with_{model_name}_features_plot.pdf', 
                            object_type = "figure", data_name = "Figure") 
            
            # Show figure
            self.maybe_show(fig=plt)
            
            # Save top features
            top_features = {feat_importances_permutation.index[i]:feat_importances_permutation[i] 
                        for i in range(len(feat_importances_permutation))}
            
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return top_features
