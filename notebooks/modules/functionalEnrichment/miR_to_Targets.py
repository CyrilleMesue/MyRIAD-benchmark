import pandas as pd  # pandas is used for data manipulation and analysis, especially when dealing with structured data like DataFrames
from modules.utils import load_json  # Custom utility function (presumably) for saving data in JSON format


class miRTARGETS:
    """
This class helps in performing operations for microRNA targets
    """

    def __init__(self, 
                 ranked_features_path:str = None,
                 num_features:int = 20
                ):
        "Initialize class attributes"

        # load miRTARBase Map
        self.miRTarBaseMap = load_json("../artifacts/miRNA_to_targets.json")

        # load ranked microRNA features
        self.top_n_features = pd.read_csv(ranked_features_path).head(num_features).microRNA.tolist()

    def run(self):
        """
        return a dictionary contain all features to a list of all their targets
        """

        # split features to versions 5p and 3p
        features_by_version = {microRNA: self.get_mirna_versions(microRNA) for microRNA in self.top_n_features}

        # Retrieve the targetS  per feature
        self.features_to_Targets, _ = self.get_targets_per_feature(self.miRTarBaseMap, features_by_version)

        return self.features_to_Targets

    def aggregate_targets(self):
        """
        combine the targets for all features.
        """
        all_targets = []

        for feature in self.top_n_features:
            targets = self.features_to_Targets[feature]

            for target in targets:
                if target not in all_targets:
                    all_targets.append(target)

        return all_targets
    
    def get_mirna_versions(self,x):
        """
        This function takes a string representing a miRNA or a complex (multiple miRNAs separated by underscores) 
        and returns a list containing all possible "-3p" and "-5p" versions for each miRNA.
        
        Parameters:
        - x (str): A string representing a single miRNA or a complex of miRNAs (joined by underscores).
        
        Returns:
        - versions (list): A list of strings containing both "-3p" and "-5p" variants for each miRNA.
        """
        
        # Split the input string by underscores to handle multiple miRNAs (if it's a complex miRNA)
        mirnas = x.split('_')  
        
        # Initialize an empty list to store all possible "-3p" and "-5p" versions of the miRNAs
        versions = []
        
        # Iterate over each miRNA in the split list
        for mirna in mirnas:
            # Check if the miRNA already contains "-3p" or "-5p" suffix
            if "3p" in mirna or "5p" in mirna:
                # Append the "-3p" or "-5p" version of the base miRNA to the list
                versions.append(mirna)
                continue
            else:
                # If the miRNA doesn't contain "-3p" or "-5p", use the miRNA name as it is
                base = mirna
    
            # Append the "-3p" version of the base miRNA to the list
            versions.append(f"{base}-3p")
            # Append the "-5p" version of the base miRNA to the list
            versions.append(f"{base}-5p")
        
        # Return the list of both "-3p" and "-5p" versions for all miRNAs
        return versions

    def get_targets_per_feature(self,miRTarBaseMap:dict = None, features_dict:dict = None):
        """
        Generates a dictionary mapping each miRNA feature to a list of its unique target genes.
    
        Parameters:
        - miRTarBaseMap (dict): A dictionary mapping miRNAs to their target genes.
        - features_dict (dict): A dictionary where the keys are miRNA features and the values are lists of miRNA forms (e.g., '-3p', '-5p').
    
        Returns:
        - targets_per_feature (dict): A dictionary mapping each miRNA feature to a list of its unique target genes.
        """
        # Create a set of all unique miRNAs from the input
        All_Unique_microRNAs = set([miRNA for m in features_dict for miRNA in features_dict[m]])
        
        # Create a dictionary mapping each miRNA to its targets (or an empty list if not found)
        miRNA_Targets = {miRNA: miRTarBaseMap[miRNA] if miRNA in miRTarBaseMap else [] for miRNA in All_Unique_microRNAs}
        
        # Initialize a dictionary to hold targets for each feature
        targets_per_feature = {feature: set() for feature in features_dict}
        
        # Populate the targets for each feature
        for feature in features_dict:
            forms = features_dict[feature]
            for form in forms:
                targets = miRNA_Targets[form]
                targets_per_feature[feature] = targets_per_feature[feature].union(set(targets))
        
        # Convert sets back to lists
        targets_per_feature = {feature: list(targets_per_feature[feature]) for feature in targets_per_feature}
        targets_per_specific_form = {feature:{microRNA: miRNA_Targets[microRNA] for microRNA in features_dict[feature]} for feature in targets_per_feature}
        return targets_per_feature, targets_per_specific_form