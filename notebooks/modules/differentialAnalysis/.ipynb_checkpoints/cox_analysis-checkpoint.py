"""
Feature selection and importance ranking using regularized logistic regression.
------------------------------------------------
Supports LASSO (L1), Ridge (L2), and ElasticNet (L1 + L2) regularized logistic regression to rank features
based on their coefficient magnitudes.

References:
- Tibshirani, R. (1997). The lasso method for variable selection in the Cox model.
  Statistics in Medicine, 16(4), 385–395. https://doi.org/10.1002/(SICI)1097-0258(19970228)16:4<385::AID-SIM380>3.0.CO;2-3
- Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net.
  Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301–320.
  
Parameters
----------
X : pandas.DataFrame, optional
    Feature matrix.
y : array-like, optional
    Target variable (binary classification).
C : float, default=1.0
    Inverse regularization strength; smaller values specify stronger regularization.
penalty : {'l1', 'l2', 'elasticnet'}, default='l1'
    Type of regularization to apply.
l1_ratio : float, optional
    ElasticNet mixing parameter (only used if penalty='elasticnet').
outdir : str, default=""
    Output directory to save feature importance plots.
feature_type : str, optional
    Label for feature type (used in plot titles).
feature_map_reverse : dict, optional
    Mapping from feature indices back to feature names.
model_finetune : bool, default=False
    Whether to perform hyperparameter tuning using cross-validation.
fine_tune_cv_nfolds : int, default=5
    Number of folds for cross-validation if fine-tuning.
scoring : str, default="f1"
    Scoring metric for model selection during hyperparameter tuning.
top_num_features_to_plot : int, default=20
    Number of top features to display in the plot.

Attributes
----------
fitted_model : sklearn.Pipeline
    Trained model pipeline.
coef_ : ndarray
    Fitted coefficients from the logistic regression model.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import set_config
from sklearn.base import clone

from modules.logger import logging
from modules.exception import CustomException
from modules.verbose_config import BaseConfigurable

set_config(transform_output="pandas")

class CoxFeatureRanker(BaseConfigurable):
    
    def __init__(self, 
                 X=None, 
                 y=None, 
                 C=1.0,
                 penalty='l1',
                 l1_ratio=0.5,
                 outdir: str = "",
                 feature_type: str = None,
                 feature_map_reverse: dict = None,
                 model_finetune: bool = False,
                 fine_tune_cv_nfolds: int = 5,
                 scoring: str = "f1",
                 top_num_features_to_plot: int = 20,
                 save_results:bool = True,
                 show_plots:bool = True
                ):

        # Store initialization parameters
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.outdir = outdir
        self.feature_type = feature_type
        self.feature_map_reverse = feature_map_reverse
        self.scoring = scoring
        self.model_finetune = model_finetune
        self.fine_tune_cv_nfolds = fine_tune_cv_nfolds
        self.top_num_features_to_plot = top_num_features_to_plot
        self.X = X
        self.y = y
        self.fitted_model = None 
        self.save_results = save_results
        self.show_plots = show_plots

        logging.info(f"Initializing Logistic Regression with penalty '{penalty}'")

        # Initialize logistic regression model with appropriate solver and regularization parameters
        model_params = {
            'penalty': self.penalty,
            'solver': 'saga' if self.penalty in ['elasticnet'] else 'liblinear',
            'C': self.C,
            'max_iter': 10000,
            'l1_ratio': self.l1_ratio if self.penalty == 'elasticnet' else None
        }
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**model_params))
        ])

    def fit(self):
        """
        Fit the logistic regression model with or without hyperparameter fine-tuning.

        Returns
        -------
        CoxFeatureRanker
            The fitted CoxFeatureRanker object.
        """
        logging.info("Fitting model and extracting coefficients.")
        try:
            if self.model_finetune:
                model = clone(self.model)
                param_grid = {
                    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]
                }
                if self.penalty == 'elasticnet':
                    param_grid["clf__l1_ratio"] = [0.1, 0.25, 0.5, 0.75, 0.9]

                kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, shuffle=True, random_state=42)
                logging.info("Fine tuning logistic regression model.")
                gs = GridSearchCV(model, param_grid, cv=kfold, scoring=self.scoring)
                gs.fit(self.X, self.y)
                model.set_params(**gs.best_params_)
                self.fitted_model = model.fit(self.X, self.y)
            else:
                self.fitted_model = self.model.fit(self.X, self.y)

            # Extract fitted coefficients
            self.coef_ = self.fitted_model.named_steps["clf"].coef_[0]

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

        return self

    def FeatureImportance(self, descending=True): 
        """
        Calculate and plot feature importance based on absolute coefficient values.

        Parameters
        ----------
        descending : bool, default=True
            Whether to sort feature importances in descending order.

        Returns
        -------
        dict
            A dictionary mapping feature names to their absolute coefficient magnitudes.
        """
        try:
            logging.info("Computing feature importances.")
            abs_coef = np.abs(self.coef_)
            feature_importance = pd.DataFrame({
                "coefficient": self.coef_,
                "abs_coefficient": abs_coef
            }, index=[self.feature_map_reverse[col] for col in self.X.columns])

            feature_importance.sort_values(by="abs_coefficient", ascending=not descending, inplace=True)

            # Select top features for plotting
            top_features = feature_importance.head(self.top_num_features_to_plot)
            colors = ["blue"] * len(top_features)

            logging.info("Plotting feature importances.")
            plt.figure(figsize=(4, 8))
            plt.barh(top_features.index, top_features["abs_coefficient"], color=colors)
            plt.xlabel("Feature Importances", fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(f"Top {self.top_num_features_to_plot} {self.feature_type}s by Importance", fontsize=15)
            plt.gca().invert_yaxis()

            # Create output directories if they don't exist 
            if self.save_results:
                os.makedirs(f"{self.outdir}/feature_importance/png", exist_ok=True)
                os.makedirs(f"{self.outdir}/feature_importance/pdf", exist_ok=True)

            # Save plots 
            mapper = lambda x: "lasso" if x=="l1" else "ridge" if x=="l2" else "elasticnet"
            self.maybe_save(plt.gcf(), f"{self.outdir}/{mapper(self.penalty)}/png/{mapper(self.penalty)}_features_plot.png", 
                    object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f"{self.outdir}/{mapper(self.penalty)}/png/{mapper(self.penalty)}_features_plot.pdf", 
                    object_type = "figure", data_name = "Figure") 
            
            # Show figure
            self.maybe_show(fig=plt)

        except Exception as e:
            logging.info(CustomException(e, sys))
            raise CustomException(e, sys)

        return feature_importance.to_dict()["abs_coefficient"]