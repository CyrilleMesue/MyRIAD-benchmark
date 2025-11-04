# Import Packages
import pandas as pd 
import os 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
import shutil
import numpy as np 
from tqdm.notebook import tqdm
from modules.benchmark_utils import * 
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import train_test_split
from modules.machineLearning.classification import Classification 
from modules.differentialAnalysis.differential_analysis import DifferentiatingFactorAnalysis 
from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker 
from modules.more_mogonet_utils import generate_and_run_training_script, generate_and_run_feat_imp_script  
from modules.utils import save_json, load_json


class BiomarkerPipeline:
    def __init__(self,  
                 data, 
                 rankers, 
                 aggregators, 
                 models, 
                 parameter_configs:dict=None,
                 labels_column_name=None, 
                 view_column_splits:dict = None, 
                 more_and_mogonet_data_dir = None, 
                 k_features:list = [], 
                 validation_datasets = None, 
                 pathway_validation_sets = None, 
                 apply_standard_scaler = False, 
                 experiment_id = "",
                ):
        """
        Initializes the pipeline.

        Parameters:
        - data: DataFrame or array-like of input features.
        - ground_truth: Set of validated biomarker features.
        - rankers: List of functions or ranker names.
        - models: List of sklearn-style classifiers.
        - cv_folds: Number of CV folds.
        """
        self.data = data 
        self.parameter_configs = parameter_configs
        self.rankers = rankers
        self.aggregators = aggregators
        self.models = models
        self.labels_column_name = labels_column_name 
        self.view_column_splits = view_column_splits 
        self.more_and_mogonet_data_dir = more_and_mogonet_data_dir
        self.results = {} 
        self.k_features = k_features  
        self.validation_datasets = validation_datasets 
        self.pathway_validation_sets = pathway_validation_sets 
        self.apply_standard_scaler = apply_standard_scaler
        self.experiment_id = experiment_id

    def prepare_dataset(self):
        """Performs KNN imputation and standard scaling on the dataset."""
        
        # Convert to DataFrame if not already
        if not isinstance(self.data, pd.DataFrame):
            self.data = pd.DataFrame(self.data)
    
        # Step 1: KNN Imputation
        imputer = KNNImputer(n_neighbors=5)  # you can adjust n_neighbors 
        sample_ids = self.data.index 
        X = self.data.drop(columns = [self.labels_column_name]) 
        column_names = X.columns
        X = pd.DataFrame(imputer.fit_transform(X),
                         columns=column_names,
                         index=sample_ids) 
        self.imputer = imputer
    
        # Step 2: Scaling 
        if self.apply_standard_scaler:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X),
                                     columns=column_names,
                                     index=self.data.index) 
            self.scaler = scaler

        self.data = pd.concat([self.data[[self.labels_column_name]], X], axis = 1)
        self.data.index = sample_ids
    
        print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features.") 

        # prepare train test data
        # Stratified split
        self.train_idx, self.test_idx = train_test_split(
                self.data.reset_index().index,
                test_size=self.parameter_configs['test_size'],
                random_state=42,
                stratify=self.data[self.labels_column_name]
        ) 

        self.train_df = self.data.iloc[self.train_idx]
        self.test_df = self.data.iloc[self.test_idx] 
        print(f"✅ Dataset prepared: {self.train_df.shape[0]} train samples, {self.test_df.shape[0]} test samples.") 

        # create splits for cross validation
        # Initialize StratifiedKFold to maintain label distribution across folds
        skf = StratifiedKFold(n_splits=self.parameter_configs['n_folds'], shuffle=True, random_state=42)
        
        # Generate train-test indices for each fold using the first omics view and labels
        self.cross_validation_indices = [(train_idx, test_idx) 
                                     for (train_idx, test_idx) in skf.split(self.data, self.data[self.labels_column_name])]

    def train_full_model(self, compute_more_mogonet = True):
        """Train classical ML models with all features using stratified CV."""
        clf = Classification(
            data = self.data, 
            labels_column = self.labels_column_name,
            n_folds = self.cross_validation_indices, 
            no_test_split = True,
            outdir =f"",
            model_list = self.models,
            verbose = False,
            save_results = False
        )
        clf.data_transfrom()
        clf.initiate_model_trainer()
        self.results["ML:Baseline"] = clf.model_report 

        # train MORE and MOGONET 
        labels =  self.data[self.labels_column_name].apply(lambda x: 1 if x == self.parameter_configs['reference_class'] else 0).values 
        view_list = list(self.view_column_splits.keys())
        
        if compute_more_mogonet:
            # run MORE cross validation
            more_results = run_more_and_mogonet_cv_pipeline(
            cross_validation_indices=self.cross_validation_indices,
            labels=labels,
            data_views=[self.data[self.view_column_splits[i]] for i in view_list],
            view_list=view_list,
            data_folder=self.more_and_mogonet_data_dir,
            script_path=f"./Code/temp_train_test_{self.experiment_id}.py",
            repo_path="../MORE",
            model_type = "more"
            ) 
            self.results["ML:Baseline"]['MORE'] = aggregate_cv_metrics(more_results)

            # run MOGONET cross validation
            
            mogonet_results = run_more_and_mogonet_cv_pipeline(
            cross_validation_indices=self.cross_validation_indices,
            labels=labels,
            data_views=[self.data[self.view_column_splits[i]] for i in view_list],
            view_list=view_list,
            data_folder=self.more_and_mogonet_data_dir,
            script_path=f"./temp_train_test_{self.experiment_id}.py",
            repo_path="../MOGONET", 
            model_type = "mogonet", 
            dim_he_list = [len(feats) for feats in self.view_column_splits.values()]
            ) 
            self.results["ML:Baseline"]['MOGONET'] = aggregate_cv_metrics(mogonet_results)

    def generate_rankings(self):
        """Apply each ranker and return ranked lists of features.""" 
        analyzer = DifferentiatingFactorAnalysis(self.data.reset_index(),
                                                 analyses = self.rankers,
                                                 labels_column = self.labels_column_name,
                                                 reference_class = self.parameter_configs['reference_class'],
                                                 sample_id_column = self.data.index.name,
                                                 outdir = '',
                                                 feature_type = self.parameter_configs['feature_type'],
                                                 test_size = self.parameter_configs['test_size'],
                                                 n_folds = self.parameter_configs['n_folds'],
                                                 scoring = self.parameter_configs['scoring'],
                                                 save_results = False,
                                                 show_plots = False, 
                                                 verbose = False
                                                ) 
        
        analyzer.run_all_analyses()
        self.results['FeatureRankng:SingleRanker'] = analyzer.feature_weights
        

    def ensemble_rankings(self):
        """Aggregate feature rankings using each aggregator."""
        
        # Load Ensemble aggregator
        ensemble_aggregator = EnsembleFeatureRanker(feature_weights = self.results['FeatureRankng:SingleRanker'],
                              feature_type = self.parameter_configs['feature_type'],
                              outdir = '',
                              decreasing_importance_cases = ["t_test", "boruta", "mannwhitneyu", "svm_rfe", 'MOGONET:Ranker', 'MORE:Ranker'],
                              save_results = False,
                              show_plots = False)
        
        # Run Aggregators 
        self.results['FeatureRankng:EnsembleRanker'] = {}
        for aggregator in tqdm(self.aggregators): 
            aggregation_method, aggregation_type = aggregator.split('_') 
            if aggregation_type == 'rank':
                 self.results['FeatureRankng:EnsembleRanker'][aggregator] = ensemble_aggregator.Rank_by_Ranks(plot = False, method = aggregation_method)
            else:
                self.results['FeatureRankng:EnsembleRanker'][aggregator] = ensemble_aggregator.Rank_by_Weights(plot = False, method = aggregation_method)

    def prepare_more_and_mogonet_data(self):
        '''''' 
        data_for_more_mogonet = self.data.copy()
        data_for_more_mogonet[self.labels_column_name] = data_for_more_mogonet[self.labels_column_name].apply(lambda x: 1 if x == self.parameter_configs['reference_class'] else 0)
        prepare_data_from_single_dataframe(
            df=data_for_more_mogonet,
            label_col=self.labels_column_name,
            view_column_splits=self.view_column_splits,
            train_idx=self.train_idx,
            test_idx=self.test_idx,
            output_dir=self.more_and_mogonet_data_dir
        ) 
        
        
    def run_more_biomarkers(self):
        # Wrapper to call MORE implementation and return predictions
        original_dir = os.getcwd()  # Save current working directory
        os.chdir('../MORE')    # Change to MORE repo directory
            
        try:
            # Run the feature importance script
            generate_and_run_feat_imp_script(
                    data_folder=self.more_and_mogonet_data_dir,
                    view_list=list(self.view_column_splits.keys()),
                    script_name=f"./Code/temp_feat_importance_{self.experiment_id}.py", 
                    feature_rankings_path=f"{self.more_and_mogonet_data_dir}/more_feature_rankings.csv"
                )
            
            # Load ranked features output from MORE
            ranked_features = pd.read_csv(f"{self.more_and_mogonet_data_dir}/more_feature_rankings.csv").feat_name.to_list() 
            self.results['MORE:Ranker'] = {item: rank for rank, item in enumerate(ranked_features, start=1)}
        finally:
            os.chdir(original_dir)  # Restore original directory
    
    def run_mogonet_biomarkers(self):
        # Wrapper to call MOGONET implementation and return predictions
        # Wrapper to call MORE implementation and return predictions
        original_dir = os.getcwd()  # Save current working directory
        os.chdir('../MOGONET')    # Change to MORE repo directory
            
        try:
            # Run the feature importance script 
            generate_and_run_feat_imp_script(
                    data_folder=self.more_and_mogonet_data_dir,
                    view_list=list(self.view_column_splits.keys()),
                    script_name=f"./temp_feat_importance_{self.experiment_id}.py", 
                    feature_rankings_path=f"{self.more_and_mogonet_data_dir}/mogonet_feature_rankings.csv"
                )
            
            # Load ranked features output from MORE
            ranked_features = pd.read_csv(f"{self.more_and_mogonet_data_dir}/mogonet_feature_rankings.csv").feat_name.to_list() 
            self.results['MOGONET:Ranker'] = {item: rank for rank, item in enumerate(ranked_features, start=1)}
        finally:
            os.chdir(original_dir)  # Restore original directory
        

    def prepare_ranked_features_for_evaluation(self, compute_more_mogonet=True):
        '''
        '''

        ranked_lists = {}  

        if compute_more_mogonet:
            ranked_lists['MORE:Ranker'] = self.results['MORE:Ranker'] 
            ranked_lists['MOGONET:Ranker'] = self.results['MOGONET:Ranker']
        for single_ranker, ranks in self.results['FeatureRankng:SingleRanker'].items():
            ranked_lists[single_ranker] = ranks
        
        for ensemble_ranker, ranks in self.results['FeatureRankng:EnsembleRanker'].items():
            ranked_lists[ensemble_ranker] = ranks 
        
        # reorder lists
        for ranker, ranks in ranked_lists.items():
            if ranker not in ["boruta", "limma", "lasso", "svm_rfe"]:
                if (ranker in ["t_test", "mannwhitneyu", 'MORE:Ranker', 'MOGONET:Ranker']) or ('rank' in ranker):
                    ascending = True
                else:
                    ascending = False
        
                ranked_lists[ranker] = rank_features(ranks, ascending=ascending) 
            else:
                ranked_lists[ranker] = list(ranks.keys()) 

        self.results['ranked_lists'] = ranked_lists
        return ranked_lists
        
    def evaluate_against_ground_truth(self):
        """
        Evaluate selected features using:
        - Precision, Recall, F1
        - TP, FP, FN counts
        - Inference score statistics (e.g., SHAP, classifier weight)
        """ 

        validation_results = {}
        
        for view, features in tqdm(self.view_column_splits.items()): 
            
            validation_results[view] = {}
            # update ranked list
            updated_ranked_list = {method: [feature for feature in ranks if feature in features] for method, ranks in self.results['ranked_lists'].items()}
            updated_ranked_list_copy = updated_ranked_list.copy()
            if len(self.validation_datasets[view]) !=0:
                
                for validation_set, val_data in self.validation_datasets[view].items():  
                    
                    if validation_set in self.pathway_validation_sets:
                        updated_ranked_list = {method: compute_pathways(feature_list) for method,feature_list in updated_ranked_list.items()}
                     
                    results_df = concat_scoring_across_cutoffs(
                        updated_ranked_list,
                        val_data,
                        method_cutoffs=self.parameter_configs['method_cutoff_values'],
                        groundtruth_cutoffs=self.parameter_configs['groundtruth_cutoff_values']
                    ) 

                    results_df = results_df.dropna()
                    validation_results[view][validation_set] = results_df   

                    updated_ranked_list = updated_ranked_list_copy.copy()

        new_results_df = pd.DataFrame()
        for view, view_data in validation_results.items():
            for validation_source, validation_source_data in view_data.items():
                validation_source_data['omicstype'] = [view]*validation_source_data.shape[0]
                validation_source_data['validationsource'] = [validation_source]*validation_source_data.shape[0] 
                new_results_df = pd.concat([new_results_df, validation_source_data], axis=0)
        
        self.results['biomarker-validation'] = new_results_df.copy()
        return new_results_df.copy()


    def cross_validate_feature_selectors(self):
        """
        Train CV models using features from:
        - Each ranker
        - Each aggregator
        - MORE and MOGONET
        """
        cross_val_results = {
            'featureSelector': [],
            'numFeatures': [],
            'modelName': [],
            'MeanAccuracy': [],
            'StdAccuracy': [],
            'MeanPrecision': [],
            'StdPrecision':[],
            'MeanRecall':[],
            'StdRecall':[],
            'MeanF1': [],
            'StdF1': [], 
            'MeanAUC': [], 
            'StdAUC': [],
            "MeanSpecificity": [],
            "StdSpecificity": [], 
            "MeanNPV": [],
            "StdNPV": [],
            "MeanLR_PLUS": [], 
            "StdLR_PLUS": [],
            "MeanLR_MINUS": [],
            "StdLR_MINUS": []
        } 

        for featureSelector, features in tqdm(self.results['ranked_lists'].items()):
            for k in tqdm(self.k_features):
                selected_features = features[:k]  
                
                topped_up_views = {view:top_up_selected_features(list(set(selected_features).intersection(set(feat_list))), feat_list, features) for view,feat_list in self.view_column_splits.items()}

                metrics = self.train_model(topped_up_views)

                for modelName, metric in metrics.items():

                    # check if dict is not empty. For MORE and MOGONET Failures
                    if len(metric) != 0:
                    
                        cross_val_results['featureSelector'].append(featureSelector) 
                        cross_val_results['numFeatures'].append(k)
                        cross_val_results['modelName'].append(modelName) 
    
                        if modelName in ['MORE', 'MOGONET']: 
                            cross_val_results['MeanAccuracy'].append(metric['accuracy']['mean'])
                            cross_val_results['StdAccuracy'].append(metric['accuracy']['std'])
                            cross_val_results['MeanF1'].append(metric['f1']['mean'])
                            cross_val_results['StdF1'].append(metric['f1']['std'])
                            cross_val_results['MeanAUC'].append(metric['roc_auc']['mean'])
                            cross_val_results['StdAUC'].append(metric['roc_auc']['std']) 
                            cross_val_results['MeanPrecision'].append(metric['precision']['mean'])
                            cross_val_results['StdPrecision'].append(metric['precision']['std'])
                            cross_val_results['MeanRecall'].append(metric['recall']['mean'])
                            cross_val_results['StdRecall'].append(metric['recall']['std'])
                            cross_val_results['MeanSpecificity'].append(metric['specificity']['mean'])
                            cross_val_results['StdSpecificity'].append(metric['specificity']['std'])
                            cross_val_results['MeanNPV'].append(metric['npv']['mean'])
                            cross_val_results['StdNPV'].append(metric['npv']['std'])
                            cross_val_results['MeanLR_PLUS'].append(metric['lr_plus']['mean'])
                            cross_val_results['StdLR_PLUS'].append(metric['lr_plus']['std'])
                            cross_val_results['MeanLR_MINUS'].append(metric['lr_minus']['mean'])
                            cross_val_results['StdLR_MINUS'].append(metric['lr_minus']['std'])
                        else:
                            cross_val_results['MeanAccuracy'].append(metric['cross_val_report']['accuracy']['mean'])
                            cross_val_results['StdAccuracy'].append(metric['cross_val_report']['accuracy']['std'])
                            cross_val_results['MeanF1'].append(metric['cross_val_report']['f1']['mean'])
                            cross_val_results['StdF1'].append(metric['cross_val_report']['f1']['std'])
                            cross_val_results['MeanAUC'].append(metric['cross_val_report']['roc_auc']['mean'])
                            cross_val_results['StdAUC'].append(metric['cross_val_report']['roc_auc']['std'])  
                            cross_val_results['MeanPrecision'].append(metric['cross_val_report']['precision']['mean'])
                            cross_val_results['StdPrecision'].append(metric['cross_val_report']['precision']['std'])
                            cross_val_results['MeanRecall'].append(metric['cross_val_report']['recall']['mean'])
                            cross_val_results['StdRecall'].append(metric['cross_val_report']['recall']['std']) 
                            cross_val_results['MeanSpecificity'].append(metric['cross_val_report']['specificity']['mean'])
                            cross_val_results['StdSpecificity'].append(metric['cross_val_report']['specificity']['std'])
                            cross_val_results['MeanNPV'].append(metric['cross_val_report']['npv']['mean'])
                            cross_val_results['StdNPV'].append(metric['cross_val_report']['npv']['std'])
                            cross_val_results['MeanLR_PLUS'].append(metric['cross_val_report']['lr_plus']['mean'])
                            cross_val_results['StdLR_PLUS'].append(metric['cross_val_report']['lr_plus']['std'])
                            cross_val_results['MeanLR_MINUS'].append(metric['cross_val_report']['lr_minus']['mean'])
                            cross_val_results['StdLR_MINUS'].append(metric['cross_val_report']['lr_minus']['std'])

        self.results['cross_val_results'] = pd.DataFrame(cross_val_results)
        return cross_val_results
        
    def train_model(self, view_column_splits, compute_more_mogonet = True):
        """Train classical ML models with all features using stratified CV.""" 
        y = self.data[[self.labels_column_name]]
        X = pd.concat([self.data[view_column_splits[i]] for i in view_column_splits.keys()
                      ], axis=1)
        new_data = pd.concat([y,X], axis = 1) 


        results = {}
        
        clf = Classification(
            data = new_data, 
            labels_column = self.labels_column_name,
            n_folds = self.cross_validation_indices,
            outdir =f"",
            model_list = self.models, 
            no_test_split = True,
            verbose = False,
            save_results = False
        )
        clf.data_transfrom()
        clf.initiate_model_trainer()
        results = clf.model_report 

        # train MORE and MOGONET 
        labels =  self.data[self.labels_column_name].apply(lambda x: 1 if x == self.parameter_configs['reference_class'] else 0).values 

        num_omics = len(view_column_splits)
        
        if compute_more_mogonet: 
            # run MORE cross validation
            more_results = run_more_and_mogonet_cv_pipeline(
            cross_validation_indices=self.cross_validation_indices,
            labels=labels,
            data_views=[self.data[view_column_splits[i]] for i in range(1,num_omics+1)],
            view_list=[1,2,3][:num_omics],
            data_folder=self.more_and_mogonet_data_dir,
            script_path=f"./Code/temp_train_test_{self.experiment_id}.py",
            repo_path="../MORE",
            model_type = "more"
            ) 
            results['MORE'] = aggregate_cv_metrics(more_results) 

            # run MOGONET cross validation
            mogonet_results = run_more_and_mogonet_cv_pipeline(
            cross_validation_indices=self.cross_validation_indices,
            labels=labels,
            data_views=[self.data[view_column_splits[i]] for i in range(1,num_omics+1)],
            view_list=[1,2,3][:num_omics],
            data_folder=self.more_and_mogonet_data_dir,
            script_path=f"./temp_train_test_{self.experiment_id}.py",
            repo_path="../MOGONET", 
            model_type = "mogonet", 
            dim_he_list = [len(feats) for feats in view_column_splits.values()]
            ) 
            results['MOGONET'] = aggregate_cv_metrics(mogonet_results) 
        return results

    def run_all(self, verbose=True, compute_more_mogonet = True):
        """Run the full pipeline in order with optional logging.""" 
    
        if verbose: print("🔧 Step 1: Preparing dataset...")
        self.prepare_dataset()

        if compute_more_mogonet: 
            if verbose: print("📂 Step 2: Preparing MORE and MOGONET input data...")
            self.prepare_more_and_mogonet_data()
        
        if verbose: print("📊 Step 3: Training full model on all features...")
        self.train_full_model()
    
        if verbose: print("📈 Step 4: Generating feature rankings...")
        self.generate_rankings()
    
        if verbose: print("🤝 Step 5: Applying ensemble ranking methods...")
        self.ensemble_rankings()

        if compute_more_mogonet:
            if verbose: print("⚙️ Step 6: Running MORE biomarker discovery...")
            self.run_more_biomarkers()
            if verbose: print("⚙️ Step 7: Running MOGONET biomarker discovery...")
            self.run_mogonet_biomarkers()
        
        if verbose: print("🧹 Step 8: Organizing ranked features for evaluation...")
        self.prepare_ranked_features_for_evaluation(compute_more_mogonet = compute_more_mogonet)
        
        if verbose: print("🧪 Step 9: Evaluating against ground truth...")
        self.evaluate_against_ground_truth()
        
        if verbose: print("🔁 Step 10: Cross-validating with feature selectors and aggregators...")
        self.cross_validate_feature_selectors()
        
        if verbose: print("✅ Pipeline completed.")


    def run_ranker_combination_experiments(self, verbose=True):
        """Run the full pipeline in order with optional logging."""
    
        if verbose: print("🔧 Step 1: Preparing dataset...")
        self.prepare_dataset()
    
        if verbose: print("📈 Step 2: Generating feature rankings...")
        self.generate_rankings()

        all_rankings = self.results['FeatureRankng:SingleRanker'].copy() 

        selected_combos = generate_random_combinations(all_rankings, sizes=(5,7,9), sample_size=100) 
        self.experimental_results = {} 
        
        for selected_combo, rankings in selected_combos.items():
            self.results['FeatureRankng:SingleRanker'] = rankings 
            
            if verbose: print("🤝 Step 3: Applying ensemble ranking methods...")
            self.ensemble_rankings()
                    
            if verbose: print("🧹 Step 8: Organizing ranked features for evaluation...")
            self.prepare_ranked_features_for_evaluation(compute_more_mogonet=False) 

            rankers = list(self.results['ranked_lists'].keys())
            
            for ranker in rankers:
                if ranker in rankings: 
                    del self.results['ranked_lists'][ranker] # remove single rankers
            
            if verbose: print("🧪 Step 9: Evaluating against ground truth...")
            self.evaluate_against_ground_truth()
            
            if verbose: print("🔁 Step 10: Cross-validating with feature selectors and aggregators...")
            self.cross_validate_feature_selectors()  

            self.experimental_results[selected_combo] = self.results.copy()
        
        if verbose: print("✅ Pipeline completed.")