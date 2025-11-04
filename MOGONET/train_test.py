""" Training and testing of the model
"""
import os
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

cuda = True if torch.cuda.is_available() else False

 
def train_test_crossval(data_folder, view_list, num_class, n_splits=5, seed=42,
                        lr_e_pretrain=1e-3, lr_e=5e-4, lr_c=1e-3, num_epoch_pretrain=500, num_epoch=1500):
    labels = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=",").astype(int)
    data_views = [np.loadtxt(os.path.join(data_folder, f"{v}_data.csv"), delimiter=",") for v in view_list] 

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc_scores, f1_scores, auc_scores = [], [], [] 

    for fold, (train_idx, test_idx) in enumerate(skf.split(data_views[0], labels)):
        print(f"\n\n=== Fold {fold+1}/{n_splits} ===")
        
        # Create temporary folder for this fold
        fold_folder = os.path.join(data_folder, f"_fold{fold+1}")
        os.makedirs(fold_folder, exist_ok=True)
    
        # Save labels
        np.savetxt(os.path.join(fold_folder, "labels_tr.csv"), labels[train_idx], fmt="%d", delimiter=",")
        np.savetxt(os.path.join(fold_folder, "labels_te.csv"), labels[test_idx], fmt="%d", delimiter=",")
    
        # Save view-specific data
        for i, view in enumerate(view_list):
            np.savetxt(os.path.join(fold_folder, f"{view}_tr.csv"), data_views[i][train_idx], delimiter=",")
            np.savetxt(os.path.join(fold_folder, f"{view}_te.csv"), data_views[i][test_idx], delimiter=",")
    
        # Call original training function
        metrics = train_test(fold_folder, view_list, num_class,
                   lr_e_pretrain, lr_e, lr_c,
                   num_epoch_pretrain, num_epoch) 
        
        acc_scores.append(metrics["accuracy"])
        f1_scores.append(metrics["f1"])
        auc_scores.append(metrics["auc"])

    print("\n=== Cross-Validation Results ===")
    print(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    final_scores = {'accuracy': acc_scores, 'f1':f1_scores, 'auc':auc_scores} 
    print(final_scores)
    return final_scores

def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


# Custom metric functions (reuse)
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def npv_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0

def lr_positive_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity / (1 - specificity) if specificity < 1 else float('inf')

def lr_negative_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return (1 - sensitivity) / specificity if specificity > 0 else float('inf')


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch,dim_he_list): 
    
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    adj_parameter = num_class
    # if (data_folder == 'ROSMAP') or ('ROSMAP' in data_folder): 
    #     adj_parameter = 2
    #     dim_he_list = [200,200,100]
    # if data_folder == 'BRCA':
    #     adj_parameter = 10
    #     dim_he_list = [400,400,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            # print("\nTest: Epoch {:d}".format(epoch))
            # if num_class == 2:
            #     print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            #     print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            #     print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            # else:
            #     print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            #     print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            #     print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            # print() 


    y_true = labels_trte[trte_idx["te"]]
    y_pred = te_prob.argmax(1)


    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, te_prob[:, 1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Additional metrics
    specificity = specificity_score(y_true, y_pred)
    npv = npv_score(y_true, y_pred)
    lr_plus = lr_positive_score(y_true, y_pred)
    lr_minus = lr_negative_score(y_true, y_pred)

    print(f"accuracy: {acc:.3f} | f1: {f1:.3f} | roc_auc: {roc:.3f} | precision: {precision:.3f} | recall: {recall:.3f} | "
          f"specificity: {specificity:.3f} | npv: {npv:.3f} | lr_plus: {lr_plus:.3f} | lr_minus: {lr_minus:.3f}")

    return {
        'accuracy': acc,
        'f1': f1,
        'roc_auc': roc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'npv': npv,
        'lr_plus': lr_plus,
        'lr_minus': lr_minus
    }