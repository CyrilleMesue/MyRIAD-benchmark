
from train_test import train_test
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":    
    num_epoch_pretrain = 500
    num_epoch = 1500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    train_test("../more_and_mogonet_data/BRCA/miRNA_and_dna_methylation_data/_fold5", [1, 2, 3], 2,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch) 
