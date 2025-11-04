# Import Packages
import requests
import pandas as pd
import sys

from modules.logger import logging
from modules.exception import CustomException

class gProfilerFunctionerEnricher:
    """
    A class to perform Gene Ontology (GO) enrichment analysis using the g:Profiler API.

    Attributes:
        parameters (dict): The query parameters used to make the POST request to the g:Profiler API.
        results (list): The results returned from the g:Profiler API after enrichment analysis.
            Each element in the list is a dictionary containing information about an enriched term.
        
    Methods:
        __init__(parameters):
            Initializes the gProfilerFunctionerEnricher object with the given parameters and performs the enrichment analysis.
        
        present_results():
            Summarizes and presents the top enriched GO terms.
    """

    def __init__(self, parameters):
        """
        Initializes the gProfilerFunctionerEnricher with the given parameters and performs the enrichment analysis.
    
        Args:
            parameters (dict): A dictionary containing the query parameters for the g:Profiler API.
            
            The parameters may include:
                - organism (str): The ID of the species to be queried (e.g., 'hsapiens' for humans).
                - query (list or dict): A list of genes or a dictionary of multiple queries.
                - sources (list): A list of data sources to use for the query (e.g., ['GO'] for Gene Ontology terms).
                    The available data sources include:
                        GO:MF - molecular function
                        GO:CC - cellular component
                        GO:BP - biological process
                        KEGG - Kyoto Encyclopedia of Genes and Genomes
                        REAC - Reactome
                        WP - WikiPathways
                        TF - Transfac
                        MIRNA - miRTarBase
                        HPA - Human Protein Atlas
                        CORUM - CORUM protein complexes
                        HP - Human Phenotype Ontology
                    An empty list is equivalent to using the full list of sources.
                    Example:
                        sources = ["GO:MF", "GO:CC", "GO:BP", "KEGG", "REAC", "WP", "TF", "MIRNA", "HPA", "CORUM", "HP"]
                - user_threshold (float): Custom significance threshold between 0 and 1.
                - all_results (bool): If True, returns results below the significance threshold.
                - ordered (bool): If True, performs an ordered query.
                - combined (bool): If True, runs queries simultaneously and combines the results.
                - numeric_ns (str): Namespace for numeric IDs, default is "ENTREZGENE".
                - measure_underrepresentation (bool): If True, returns under-represented functional terms.
                - significance_threshold_method (str): Multiple testing correction method. Default is 'g_SCS'. Options include 'bonferroni' and 'fdr'.
                - no_evidences (bool): If True, skips lookup for evidence codes, speeding up queries.
                - no_iea (bool): If True, excludes electronically annotated GO terms.
                - domain_scope (str): Scope of the domain. Default is 'annotated'. Options include 'known', 'custom', 'custom_annotated'.
                    If 'custom' or 'custom_known' is used, the 'background' parameter must be populated.
                - background (list): A list of gene IDs used as the statistical background. Required if 'domain_scope' is set to 'custom'.
                - output (str): The format of the output. Default is 'json'.
                - highlight (bool): If True, adds a 'highlighted' column to the results.
    
        Raises:
            Exception: If the POST request fails or if the API returns an error.
        """

        try:
            logging.info(f"Fetching GO Terms based on Parameters: {parameters}")
            r = requests.post(
                url='https://biit.cs.ut.ee/gprofiler/api/gost/profile/',
                json=parameters,
                headers={'User-Agent': 'FullPythonRequest'}
            )
            self.results = r.json().get('result', [])
            self.tabular_results = None

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

    def results_table(self):
        """
        Presents a summary of the enrichment analysis results as a table

        Returns:
            pa.DataFrame
        """

        try:
            logging.info("Converting Results to Tabular Format")
            self.tabular_results = pd.DataFrame.from_dict(self.results)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return self.tabular_results


    def results_filter(self, 
                       p_value: float = 0.05,
                       precision: float = 0.0,
                       recall: float = 0.0,
                       significant: bool = True) -> None:
        """
        Filters the tabular results based on specified thresholds for p-value, precision, recall, and significance.
    
        Args:
            p_value (float, optional): The maximum p-value for the results to be included. Default is 0.05.
            precision (float, optional): The minimum precision value for the results to be included. Default is 0.0.
            recall (float, optional): The minimum recall value for the results to be included. Default is 0.0.
            significant (bool, optional): If True, only includes results marked as significant. Default is True.
    
        Returns:
            None: The method updates the tabular results in place based on the filtering criteria.
        """

        try:
            logging.info(f"Filtering Results by p_value < {p_value}, precision >= {precision}, recall >= {recall} and by significant: {significant}" )
            if type(self.tabular_results) == type(None):
                self.results_table()
            self.tabular_results = self.tabular_results[
                (self.tabular_results.p_value < p_value) & 
                (self.tabular_results.precision >= precision) &
                (self.tabular_results.recall >= recall) & 
                (self.tabular_results.significant == significant)
            ]
            self.tabular_results.reset_index(inplace= True, drop = True)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)
            
        return self.tabular_results

    def results_sort(self, sort_by: dict) -> None:
        """
        Sorts the tabular results based on specified columns and order.
    
        Args:
            sort_by (dict): A dictionary specifying the columns to sort by and their respective order.
                Example format: {"by": ["p_value", "precision"], "ascending": [True, False]}.
                    - "by" (list of str): The columns to sort by.
                    - "ascending" (list of bool): The sort order for each column; True for ascending, False for descending.
    
        Returns:
            None
        """

        try:
            logging.info(f"Sorting Results in the Order {sort_by}")
            if type(self.tabular_results) is type(None):
                self.results_table()
                
            self.tabular_results = self.tabular_results.sort_values(
                by=sort_by["by"],
                ascending=sort_by["ascending"]
            )
            self.tabular_results.reset_index(inplace= True, drop = True)

        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return self.tabular_results

    
    def remove_parent_terms(self) -> None:
        """
        Filters the tabular results to retain only the most specific terms by eliminating parent terms.
    
        This method checks if any term in the results has a child term in the list.
        If a term has a child term present, it is considered a parent and is removed from the results.
        The most specific term is kept, and all of its parents/ancestors are removed from the list.
    
        Returns:
            None: The method updates `self.tabular_results` in place, retaining only the most specific terms.
        """

        try:
            logging.info("Removing Parent Terms")
            if self.tabular_results is None:
                self.results_table()
    
            if len(self.tabular_results) == 0:
                return self.tabular_results
                        
            # Create a set to hold the most specific terms
            specific_terms = set(self.tabular_results['native'])
            
            # Iterate over each term to remove its parents if a more specific term exists
            for i, row in self.tabular_results.iterrows():
                # For each term, check its parent terms and remove them from the specific_terms set
                for parent_term in row['parents']:
                    if parent_term in specific_terms:
                        specific_terms.remove(parent_term)
            
            # Filter the DataFrame to keep only the most specific terms
            self.tabular_results = self.tabular_results[self.tabular_results['native'].isin(specific_terms)]
            self.tabular_results.reset_index(inplace= True, drop = True)
        except Exception as e:
            logging.info(CustomException(e,sys))
            raise CustomException(e, sys)

        return self.tabular_results
