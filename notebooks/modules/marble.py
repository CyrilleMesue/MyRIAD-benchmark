"""
Main Script for Running Feature Ranking and Evaluation with MARBLE

This script sets up and runs biomarker selection and evaluation using the MARBLE framework.
It includes loading necessary packages, utilities for saving outputs, and initializing the MARBLE pipeline.

Author: Cyrille Mesue NJUME
Date: 11 June 2025
"""

# Imports
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt 
import sys

from modules.machineLearning.classification import Classification
from modules.differentialAnalysis.differential_analysis import DifferentiatingFactorAnalysis
from modules.featureSelection.TA import ThresholdAggregator
from modules.featureSelection.MRA import MedRankAggregator
from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from modules.logger import logging
from modules.exception import CustomException
from modules.verbose_config import BaseConfigurable

warnings.filterwarnings('ignore')


class MARBLE(BaseConfigurable):
    """
    MARBLE: A pipeline for Multi-Aspect Ranking for Biomarker Learning and Evaluation.

    This class performs:
    - Ranker execution (LIMMA, Boruta, etc.)
    - Feature aggregation via rank or weight-based methods
    - Biomarker prediction and classification model training
    """

    def __init__(
        self,
        data: pd.DataFrame,
        custom_test_data: tuple = None,
        labels_column: str = "Diagnosis",
        reference_class: str = "Control",
        sample_id_column: str = "Sample ID",
        outdir: str = "output",
        mode: str = "classification",
        feature_type: str = "microRNA",
        test_size: float = 0.2,
        lime_global_explanation_sample_num: int = 10,
        ranker_model_finetune: bool = False,
        classifier_finetune: bool = False,
        n_folds: int = 10,
        scoring: str = "f1",
        rankers=None,
        k_features: int = 50,
        verbose: bool = True,
        aggregation_type: str = 'weight',
        aggregation_method: str = 'mean',
        model_list: list = ["Logistic Regression", "Decision Tree"],
        request_cross_val_report:bool = True
    ):
        """
        Initializes the MARBLE pipeline.

        Parameters:
        - data: Full dataset including features, sample IDs, and labels
        - custom_test_data: Optional tuple (X_test, y_test) to bypass internal train-test split
        - aggregation_type: 'rank' or 'weight'
        - aggregation_method: Aggregation function (e.g., 'mean', 'rra', 'ta')
        """
        try:
            assert aggregation_type in ['rank', 'weight'], "aggregation_type must be 'rank' or 'weight'"
            if aggregation_type == 'rank':
                assert aggregation_method in ['mean', 'min', 'median', 'rra', 'geom.mean', 'stuart', 'mra']
            else:
                assert aggregation_method in ['mean', 'max', 'median', 'geom.mean', 'ta']

            self.data = data
            self.labels_column = labels_column
            self.reference_class = reference_class
            self.sample_id_column = sample_id_column
            self.outdir = outdir
            self.mode = mode
            self.feature_type = feature_type
            self.test_size = test_size
            self.lime_global_explanation_sample_num = lime_global_explanation_sample_num
            self.ranker_model_finetune = ranker_model_finetune
            self.classifier_finetune = classifier_finetune
            self.n_folds = n_folds
            self.scoring = scoring
            self.rankers = rankers or []
            self.k_features = k_features
            self.verbose = verbose
            self.aggregation_type = aggregation_type
            self.aggregation_method = aggregation_method
            self.model_list = model_list
            self.trained_model = None 
            self.request_cross_val_report = request_cross_val_report

            self.X_test, self.y_test = custom_test_data if custom_test_data else (None, None)
        except Exception as e:
            logging.error("Error initializing MARBLE")
            raise CustomException(e, sys)

    def run_rankers(self):
        """Executes all specified rankers and stores their outputs."""
        try:
            analyzer = DifferentiatingFactorAnalysis(
                self.data,
                analyses=self.rankers,
                labels_column=self.labels_column,
                reference_class=self.reference_class,
                sample_id_column=self.sample_id_column,
                outdir=self.outdir,
                feature_type=self.feature_type,
                test_size=self.test_size,
                lime_global_explanation_sample_num=self.lime_global_explanation_sample_num,
                shap_model_finetune=self.ranker_model_finetune,
                lime_model_finetune=self.ranker_model_finetune,
                cox_model_finetune=self.ranker_model_finetune,
                n_folds=self.n_folds,
                scoring=self.scoring,
                verbose=self.verbose,
                save_results=False,
                show_plots=False
            )
            analyzer.run_all_analyses()
            self.feature_weights = analyzer.feature_weights
            return self.feature_weights
        except Exception as e:
            logging.error("Failed to run rankers")
            raise CustomException(e, sys)

    def aggregate_features(self):
        """Aggregates features using the specified method."""
        try:
            logging.info(f"Aggregating Features with {self.aggregation_method.capitalize()} {self.aggregation_type.capitalize()} Method")
            length = 110
            self.print("=" * length)
            self.print(f"Aggregating Features with {self.aggregation_method.capitalize()} {self.aggregation_type.capitalize()} Method")
            self.print("=" * length)

            self.ensemble_aggregator = EnsembleFeatureRanker(
                feature_weights=self.feature_weights,
                feature_type=self.feature_type,
                outdir=self.outdir,
                num_top_features=self.k_features,
                decreasing_importance_cases=["t_test", "boruta", "mannwhitneyu", "svm_rfe"],
                save_results=False,
                show_plots=False
            )

            if self.aggregation_type == "rank":
                self.aggregation_ranking = self.rank_based_aggregation()
            elif self.aggregation_type == "weight":
                self.aggregation_ranking = self.weight_based_aggregation()

            self.print("=" * length)
            self.print(f"{self.aggregation_type.capitalize()} Aggregation Complete")
            self.print("=" * length)
            return self.aggregation_ranking
        except Exception as e:
            logging.error("Feature aggregation failed")
            raise CustomException(e, sys)

    def rank_based_aggregation(self) -> list[str]:
        """Aggregate features using rank-based method (e.g., mean rank, RRA)."""
        return self.ensemble_aggregator.Rank_by_Ranks(plot=False, method=self.aggregation_method)

    def weight_based_aggregation(self) -> list[str]:
        """Aggregate features using weight-based method (e.g., mean p-value, TA)."""
        return self.ensemble_aggregator.Rank_by_Weights(plot=False, method=self.aggregation_method)

    def predict_biomarker(self, k_features: int = 50, with_weights:bool = False) -> list[str]:
        """Select top k features based on aggregated ranks or weights.

        This method returns the top-k features along with their aggregated scores.
        - If a **rank-based** aggregator is used, lower aggregated scores indicate more important features.
        - If a **weight-based** aggregator is used, higher aggregated scores indicate more important features.
        """

        ascending = True if self.aggregation_type == 'rank' else False

        def sorted_keys_by_value(d, ascending=True):
            return [k for k, v in sorted(d.items(), key=lambda x: x[1], reverse=not ascending)]
        
        selected_features = sorted_keys_by_value(self.aggregation_ranking, ascending=ascending)[:k_features]

        if with_weights:
            return {feature: self.aggregation_ranking[feature] for feature in selected_features} 
            
        return selected_features

    def train_models(self):
        """Train classification models using selected features."""
        try: 
            self.selected_features = self.predict_biomarker(k_features=self.k_features) 
            data = self.data[self.selected_features + [self.labels_column]]

            if self.X_test is not None:
                X_test = self.X_test[self.selected_features]
            else:
                X_test = None

            clf = Classification(
                data=data,
                X_test=X_test,
                y_test=self.y_test,
                labels_column=self.labels_column,
                n_folds=self.n_folds,
                test_size=self.test_size,
                outdir=self.outdir,
                param_finetune=self.classifier_finetune,
                finetune_fraction=1.0,
                save_best_model=True,
                standard_scaling=True,
                save_data_transformer=True,
                save_label_encoder=True,
                model_list=self.model_list,
                verbose=self.verbose,
                save_results=False, 
                request_cross_val_report = self.request_cross_val_report
            )

            clf.data_transfrom()
            self.data_transformer = clf.data_transformer 
            self.label_encoder = clf.label_encoder
            clf.initiate_model_trainer()

            self.best_model = clf.best_model
            self.best_model_name = clf.best_model_name
            self.model_classification_report = clf.model_report
            self.trained_model = self.best_model  # Ensure predict() works
        except Exception as e:
            logging.error("Model training failed")
            raise CustomException(e, sys)

    def fit(self):
        """Run rankers, aggregate features, and train models."""
        try:
            self.run_rankers()
            self.aggregate_features()
            self.train_models()
        except Exception as e:
            logging.error("MARBLE pipeline execution failed")
            raise CustomException(e, sys)

    def predict(self, X: pd.DataFrame):
        """Predict outcomes using the trained model."""
        try:
            if self.trained_model is None:
                raise Exception("Model not trained yet. Call `fit()` first.") 

            X = X[self.selected_features]
            X.columns = [f"Feature_{i}" for i, feature in enumerate(self.selected_features)]
            
            transformed_X = self.data_transformer.transform(X)
            return self.label_encoder.inverse_transform(self.trained_model.predict(transformed_X))
        except Exception as e:
            logging.error("Prediction failed")
            raise CustomException(e, sys)
