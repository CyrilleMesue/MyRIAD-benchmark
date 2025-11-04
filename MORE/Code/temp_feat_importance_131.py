
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    data_folder = '../more_and_mogonet_data/ROSMAP/miRNA_and_gene_expression_and_dna_methylation_data'
    model_folder = os.path.join(data_folder, 'models')
    view_list = [1, 2, 3]
    num_class = 2

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep+1)), 
                                    view_list, num_class)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(featimp_list_list, feature_rankings_path="../more_and_mogonet_data/ROSMAP/miRNA_and_gene_expression_and_dna_methylation_data/more_feature_rankings.csv")
