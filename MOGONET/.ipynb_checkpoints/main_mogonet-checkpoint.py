""" Example for MOGONET classification
"""
from train_test import train_test, train_test_crossval

if __name__ == "__main__":    
    data_folder = 'ROSMAP2'
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
    if  (data_folder == 'ROSMAP') or (data_folder == 'ROSMAP2'):
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    
    # train_test(data_folder, view_list, num_class,
    #            lr_e_pretrain, lr_e, lr_c, 
    #            num_epoch_pretrain, num_epoch)              


    train_test_crossval(data_folder, view_list, num_class, n_splits=10, seed=42,
                        lr_e_pretrain=lr_e_pretrain, lr_e=lr_e, lr_c=lr_c, num_epoch_pretrain=num_epoch_pretrain, num_epoch=num_epoch)