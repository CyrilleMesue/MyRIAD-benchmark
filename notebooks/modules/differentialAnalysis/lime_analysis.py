# Load general packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import contextlib

# Load sklearn packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import set_config

# Load specialized packages
import lime
import lime.lime_tabular
import itertools

# Load Custom Modules
from modules.utils import save_json, load_json, getparams
from modules.logger import logging
from modules.exception import CustomException 
from modules.verbose_config import BaseConfigurable

# Set sklearn configuration
set_config(transform_output="pandas")


class LIME_Analysis(BaseConfigurable):
    """
    This class performs LIME analysis on a dataset and plots the explainability 
    of top differentiating features.

    Attributes:
    -----------
    X : pd.DataFrame, optional
        The feature matrix.
    y : pd.Series, optional
        The target variable.
    class_names : list, optional
        The names of the classes.
    mode : str, optional, default="classification"
        The mode for LIME ("classification" or "regression").
    random_samples : dict, optional
        Random samples for each class.
    outdir : str, optional, default=""
        Output directory for saving plots.
    feature_type : str, optional
        Type of feature being analyzed (e.g., continuous or categorical).
    global_explanation_sample_num : int, optional, default=10
        Number of samples for global explanations.
    feature_map_reverse : dict, optional
        Mapping of features to their original names or formats.
    model_finetune : bool, optional, default=False
        Whether to fine-tune the model for better LIME analysis.
    fine_tune_cv_nfolds : int, optional, default=5
        Number of cross-validation folds for fine-tuning.
    scoring : str, optional, default="f1"
        Scoring metric used for fine-tuning the model.
    top_num_features_to_plot : int, optional, default=20
        The number of features to display per plot
    """

    def __init__(self, 
                 X=None, 
                 y=None, 
                 class_names: list = None, 
                 mode: str = "classification", 
                 random_samples: dict = None,
                 outdir: str = "",
                 feature_type: str = None,
                 global_explanation_sample_num: int = 10,
                 feature_map_reverse:dict = None,
                 model_finetune:bool = False,
                 fine_tune_cv_nfolds:int = 5,
                 scoring:str = "f1",
                 top_num_features_to_plot:int = 20,
                 save_results:bool = True,
                 show_plots:bool = True
                ):
        """
        Initialize the LIME_Analysis class.

        Parameters:
        -----------
        X : pd.DataFrame, optional
            The feature matrix to be analyzed.
        y : pd.Series, optional
            The target variable corresponding to the feature matrix.
        class_names : list, optional
            The names of the target classes (for classification tasks).
        mode : str, optional, default="classification"
            Specifies the type of task: either "classification" or "regression".
        random_samples : dict, optional
            Dictionary containing random samples to be used for each class.
        outdir : str, optional, default=""
            The directory where the output plots and explanations will be saved.
        feature_type : str, optional
            Specifies the type of features (e.g., continuous or categorical).
        global_explanation_sample_num : int, optional, default=10
            The number of samples to use when generating global explanations.
        feature_map_reverse : dict, optional
            Dictionary to reverse map features for interpretation.
        model_finetune : bool, optional, default=False
            If True, the model will be fine-tuned before LIME analysis.
        fine_tune_cv_nfolds : int, optional, default=5
            The number of cross-validation folds for model fine-tuning.
        scoring : str, optional, default="f1"
            The scoring metric to be used for model fine-tuning (e.g., "accuracy", "f1").
        top_num_features_to_plot : int, optional, default = 20
            The number of features to display per plot
        """
        self.X = X
        self.y = y
        self.class_names = class_names
        self.mode = mode
        self.random_samples = random_samples
        self.outdir = outdir
        self.feature_type = feature_type
        self.global_explanation_sample_num = global_explanation_sample_num
        self.feature_map_reverse = feature_map_reverse
        self.scoring = scoring
        self.model_finetune = model_finetune
        self.fine_tune_cv_nfolds = fine_tune_cv_nfolds
        self.top_num_features_to_plot = top_num_features_to_plot
        self.fitted_model = None 
        self.save_results = save_results
        self.show_plots = show_plots


    def fit(self):
        """
        Fit the Random Forest Model and Initiate LIME Explanations.
        """

        try:

            if self.model_finetune: # Check if model fine-tuning is enabled
                model = RandomForestClassifier() # initialize model
                params = getparams()["Random Forest"]
                kfold = StratifiedKFold(n_splits=self.fine_tune_cv_nfolds, random_state=42, shuffle = True)
    
                logging.info("Fine tuning RandomForestClassifier") # Log the start of model fine-tuning process
                gs = GridSearchCV(model,params,cv=kfold, scoring = self.scoring)
                gs.fit(self.X,self.y) #  fit grid search object for parameter tunning
                model_best_params = gs.best_params_
                model.set_params(**model_best_params) # Update model with optimal parameters
                self.fitted_model = model.fit(self.X, self.y)
            else:
                self.fitted_model = RandomForestClassifier(max_depth=4, random_state=32).fit(self.X, self.y)
    
            # initialize LIME explainer
            self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.X.values,
                                                       feature_names=list(map(lambda x: self.feature_map_reverse[x],self.X.columns.values)),
                                                       discretize_continuous=True,
                                                       class_names=self.class_names[::-1],
                                                       mode=self.mode,
                                                       verbose=True,
                                                       random_state=32)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
        
    def _check_fit(self):
        """
        Checks if the model is fitted and the LIME explainer is initialized.
        If not, it calls the fit method to train the model.
        """
        try:
            if not hasattr(self, 'explainer') or self.explainer is None: # Ensure LIME values are computed before plotting
                logging.info("Training RandomForestClassifier for LIME Analysis")
                self.fit()

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
            
    def explain_samples(self):
        """
        Explain samples using LIME and plot the local explanation for the top n (top_num_features_to_plot) features.
        """
        try:
            self._check_fit()
    
            logging.info("Computing LIME Per Sample Explanations and Plotting") 
            #Explaining a random class 0 sample using top n(top_num_features_to_plot) features
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                exp0 = self.explainer.explain_instance(self.X.values[self.random_samples[self.class_names[0]],:],
                                                       self.fitted_model.predict_proba, num_features=self.top_num_features_to_plot)
            #Plot local explanation
            plt2 = exp0.as_pyplot_figure()
            plt.title(f"Local explanation for class {self.class_names[0]} on an {self.class_names[0]} Sample", x=0.3)
            plt2.tight_layout()
            
            # Save the plot as a PNG file
            logging.info("Saving Plot 1")
            self.maybe_save(plt2, f'{self.outdir}/png/lime_local_explanation_plot_{self.class_names[0]}.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt2, f'{self.outdir}/png/lime_local_explanation_plot_{self.class_names[0]}.pdf', 
                       object_type = "figure", data_name = "Figure")

            # Show figure
            self.maybe_show(fig=plt)
            
            # plot lime explainer htmls
            if self.show_plots:
                exp0.show_in_notebook(show_table=True)
    
            #Explaining a random class 1 sample using top n (top_num_features_to_plot) features
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                exp1 = self.explainer.explain_instance(self.X.values[self.random_samples[self.class_names[1]],:],
                                                       self.fitted_model.predict_proba, num_features=self.top_num_features_to_plot)
            #Plot local explanation
            plt2 = exp1.as_pyplot_figure()
            plt.title(f"Local explanation for class {self.class_names[0]} on an {self.class_names[1]} Sample", x=0.3)
            plt2.tight_layout()
            
            # Save the plot as a PNG file
            logging.info("Saving Plot 2")
            self.maybe_save(plt2, f'{self.outdir}/png/lime_local_explanation_plot_{self.class_names[1]}.png', 
                       object_type = "figure", data_name = "Figure")
            self.maybe_save(plt2, f'{self.outdir}/png/lime_local_explanation_plot_{self.class_names[1]}.pdf', 
                       object_type = "figure", data_name = "Figure")           

            # Show figure
            self.maybe_show(fig=plt)
    
            # plot lime explainer htmls
            if self.show_plots:
                exp1.show_in_notebook(show_table=True)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def get_lime_explanations(self):
        """
        Generate LIME explanations for all samples in the dataset.

        Returns:
        - explanations (list): A list of feature importance pairs for each sample.
        """

        try:
            self._check_fit()
            logging.info(f"Computing LIME Explanations from {self.global_explanation_sample_num} samples") 
            # Define the number of samples you want from each class
            n0 = min(self.global_explanation_sample_num, sum(self.y==0))
            n1 = min(self.global_explanation_sample_num, sum(self.y==1))
            
            # Separate the data into two classes
            class_0_indices = np.where(self.y == 0)[0]
            class_1_indices = np.where(self.y == 1)[0]
            
            # Randomly sample n indices from each class
            class_0_sample = np.random.choice(class_0_indices, n0, replace=False)
            class_1_sample = np.random.choice(class_1_indices, n1, replace=False)
            
            # Combine the sampled indices
            sample_indices = np.concatenate([class_0_sample, class_1_sample])
            
            # Create the new X and y
            X_new = self.X.iloc[sample_indices]
            
            # Optionally, shuffle the new data
            shuffled_indices = np.random.permutation(X_new.shape[0])
            X_new = X_new.iloc[shuffled_indices]
    
            explanations = []
            for i in range(X_new.shape[0]):
                exp = self.explainer.explain_instance(X_new.iloc[i], self.fitted_model.predict_proba,
                                                      num_features=len(X_new.columns));
                explanations.append(exp.as_list())

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return explanations   


    def aggregate_explanations(self):
        """
        Aggregate the LIME explanations to calculate mean and standard deviation of feature importance.

        Returns:
        - feature_means (dict): Mean importance of each feature.
        - feature_stds (dict): Standard deviation of importance for each feature.
        """

        try:
            self._check_fit()
            logging.info(f"Aggregating LIME explanations from {self.global_explanation_sample_num} samples")
            # Suppress print statements
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull):
                    # Your code that you want to suppress output from
                    explanations = self.get_lime_explanations()
            
            feature_importances = {}
            for explanation in explanations:
                for feature, importance in explanation:
                    # clean features
                    if feature in feature_importances:
                        feature_importances[feature.split(">")[0].split("<=")[0].split("<")[-1].strip()].append(importance)
                    else:
                        feature_importances[feature.split(">")[0].split("<=")[0].split("<")[-1].strip()] = [importance]
        
            feature_means = {k: np.mean(v) for k, v in feature_importances.items()}
            feature_stds = {k: np.std(v) for k, v in feature_importances.items()}

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return feature_means, feature_stds


    def limeFeatureImportance(self):
        """
        Plot and return the feature importance from LIME explanations.

        Returns:
        - top_features (dict): Dictionary of top features with their importance.
        """
        try:
            self._check_fit()
            logging.info(f"Computing LIME Feature Importances from {self.global_explanation_sample_num} samples")
            feature_means, feature_stds = self.aggregate_explanations()
    
            # Convert feature means and stds to a DataFrame
            feature_df = pd.DataFrame(list(feature_means.items()), columns=['Feature', 'Mean Importance'])
            feature_df['Std Importance'] = feature_stds.values()
            
            # Sort by absolute mean importance and select top 30 features
            feature_df['Abs Mean Importance'] = feature_df['Mean Importance'].abs()
            top_n_features = feature_df.sort_values(by='Abs Mean Importance', ascending=False).head(self.top_num_features_to_plot)
            
            # Set colors for positive and negative mean importance
            colors = top_n_features['Mean Importance'].apply(lambda x: 'green' if x > 0 else 'red')
            
            # Plot the original mean importance values with error bars
            plt.figure(figsize=(4, 8))
            bars = plt.barh(top_n_features['Feature'], top_n_features['Mean Importance'], 
                            xerr=top_n_features['Std Importance'], color=colors)
            plt.title(f'LIME Feature Importance of Top Differentiating {self.feature_type}s',fontsize=15, loc='center', pad=15)
            plt.gca().invert_yaxis()
            plt.xlabel('Mean Importance', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
    
            # save
            logging.info("Saving LIME Feature Importance Plots") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/lime_summary_plot.png', 
                            object_type = "figure", data_name = "Figure") 
            self.maybe_save(plt.gcf(), f'{self.outdir}/png/lime_summary_plot.pdf', 
                            object_type = "figure", data_name = "Figure") 
            
            # Show figure
            self.maybe_show(fig=plt)
    
            feature_df = feature_df.sort_values(by = "Abs Mean Importance", ascending = False)
            top_features = {feature_df.Feature[i] : feature_df["Abs Mean Importance"][i] for i in range(len(feature_df))}

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
        return top_features


