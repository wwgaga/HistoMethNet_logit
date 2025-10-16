import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
from scipy.stats import pearsonr
from models.rrt import RRTMIL

import pandas as pd
import argparse
from utils.utils import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def get_rrt_encoder(ckpt_path, args, device='cuda'):
    # parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    
    # args = parser.parse_args()
    
    args.all_shortcut=True 
    args.crmsa_mlp=True 
    args.epeg_k=13 
    args.crmsa_k=3 
    args.crmsa_heads=1 

    model_dict = {
            'input_dim': args.embed_dim,
            'n_classes': args.n_classes,
            'dropout': args.drop_out,
            'act': args.act,
            'region_num': args.region_num,
            'pos': args.pos,
            'pos_pos': args.pos_pos,
            'pool': args.pool,
            'peg_k': args.peg_k,
            'drop_path': args.drop_path,
            'n_layers': args.n_trans_layers,
            'n_heads': args.n_heads,
            'attn': args.attn,
            'da_act': args.da_act,
            'trans_dropout': args.trans_drop_out,
            'ffn': args.ffn,
            'cell_property': args.cell_property,
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'min_region_num': args.min_region_num,
            'qkv_bias': args.qkv_bias,
            'epeg_k': args.epeg_k,
            'epeg_2d': args.epeg_2d,
            'epeg_bias': args.epeg_bias,
            'epeg_type': args.epeg_type,
            'region_attn': args.region_attn,
            'peg_1d': args.peg_1d,
            'cr_msa': args.cr_msa,
            'crmsa_k': args.crmsa_k,
            'all_shortcut': args.all_shortcut,
            'crmsa_mlp':args.crmsa_mlp,
            'crmsa_heads':args.crmsa_heads,
         }
    model = RRTMIL(**model_dict)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)  # Load the checkpoint
    # print(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)            # Pass the loaded state_dict
    model.eval()
    model.to(device)
    return model


def eval(dataset, args, ckpt_path, device):
    # model = initiate_model(args, ckpt_path)
    model = get_rrt_encoder(ckpt_path, args, device=device)
    print('Init Loaders')
    # loader = get_simple_loader(dataset)
    print('num of testing samples:')
    print(len(dataset))
    loader = get_split_loader(dataset,  testing = args.testing)
    print('num of testing samples:')
    print(len(loader))
    
    patient_results, val_error, val_auc, _, mae_value, r_value = summary_rrt(model, loader, args.n_classes, cell_property=args.cell_property)

    print('Val error: {:.4f}, ROC AUC: {:.4f}, Pearson r score cell_type: {:.4f}, MAE score cell type: {:.4f}'.format(val_error, val_auc, r_value, mae_value))

    # patient_results, test_error, auc, df, _ = summary(model, loader, args)
    # print('test_error: ', test_error)
    # print('auc: ', auc)
    return model, patient_results, val_error, val_auc, mae_value, r_value

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger



def summary_rrt(model, loader, n_classes, cell_property='cell_type'):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    r2_score_stem_immune_list = []
    r2_score_cell_type_list = []

    real_stem_ratios = []
    pred_stem_ratios = []

    real_immune_ratios = []
    pred_immune_ratios = []

    real_cell_ratios = []
    pred_cell_ratios = []

    r2 = R2Score().to(device)
    # print('length of loader')
    # print(len(loader))
    for batch_idx, data in enumerate(loader):
        # print(batch_idx)
        data, label, stem, immune, eight_cell_type, min_max_dict = data[0].to(device), \
        data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5]    
        
        if cell_property == 'cell_type': 
            cell_type_percentages = eight_cell_type
        elif cell_property == 'stem': 
            cell_type_percentages = stem
        elif cell_property == 'immune': 
            cell_type_percentages = immune
        # print(data.shape)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            instance_dict = model(data)
            # logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        
        # stem_immune_prob = instance_dict['stem_immune_prob'] 
        cell_type_prob = instance_dict['cell_type_prob']
        # Y_prob = torch.softmax(logits,dim=-1).cpu().squeeze().numpy()
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #     # acc_logger.log(Y_hat, label)
        # acc_logger.log(Y_hat, label)
        # probs = Y_prob
        
        # stem_prob = stem_immune_prob[0][0].detach().cpu().numpy()
        # pred_stem_ratios.append(stem_prob)
        # # print(stem_prob)
        # # print(stem_immune)
        # stem_ratio = stem_immune[0].detach().cpu().numpy()
        # real_stem_ratios.append(stem_ratio)
        
        # immune_prob = stem_immune_prob[0][1].detach().cpu().numpy()
        # immune_label = stem_immune[1].detach().cpu().numpy()

        # pred_immune_ratios.append(immune_prob)
        # real_immune_ratios.append(immune_label)

        pred_cell_type_prob = cell_type_prob.detach().cpu().numpy()
        pred_cell_ratios.append(pred_cell_type_prob)
        real_cell_ratios.append(cell_type_percentages.detach().cpu().numpy())
        probs = 0.0
        all_probs[batch_idx] =  probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        # error = calculate_error(Y_hat, label)
        # test_error += error

    test_error /= len(loader)

    # if n_classes == 2:
    #     auc = roc_auc_score(all_labels, all_probs[:, 1])
    #     aucs = []
    # else:
    #     aucs = []
    #     binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
    #     for class_idx in range(n_classes):
    #         if class_idx in all_labels:
    #             fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
    #             aucs.append(calc_auc(fpr, tpr))
    #         else:
    #             aucs.append(float('nan'))

    #     auc = np.nanmean(np.array(aucs))
    # mean_r2_score_stem_immune = np.mean(np.array(r2_score_stem_immune_list))
    # mean_r2_score_cell_type = np.mean(np.array(r2_score_cell_type_list))
    # print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, r2 score stem immune: {:.4f}, r2 score cell_type: {:.4f}'.format(val_loss, val_error, auc, mean_r2_score_stem_immune, mean_r2_score_cell_type))
    # mean_r2_score_stem = r2_score(np.array(real_stem_ratios), np.array(pred_stem_ratios))
 
    # mean_r2_score_immune = r2_score(np.array(real_immune_ratios), np.array(pred_immune_ratios))
    mean_r2_score_stem = 0.0
    mean_r2_score_immune = 0.0
    auc = 0.0
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute MAE for other cell types
    real_cell_ratios = np.vstack(real_cell_ratios).flatten()
    pred_cell_ratios = np.vstack(pred_cell_ratios).flatten()
   
    mae_value = torch.abs(torch.tensor(real_cell_ratios) - torch.tensor(pred_cell_ratios)).mean(dim=0).item()
    # print("MAE for cell types propotions:", mae_value)
    
    # mean_r2_score_cell_type = r2_score(np.vstack(real_cell_ratios), np.vstack(pred_cell_ratios))

    # Compute Pearson's r
    r_value, p_value = pearsonr(pred_cell_ratios, real_cell_ratios)
    
    # print(f"Overall Pearson's r across all cell types and samples: {r_value:.4f}")
    

    # if writer:
    #     # Replace SummaryWriter.add_scalar with wandb.log
    #     wandb.log({
    #         'test_error': test_error,
    #         'test_auc': auc,
    #         'test_mae': mae_value.tolist(),
    #         'test_pearson_r': r_value,
    #         'test_r2_cell_type': mean_r2_score_cell_type
    #     })

    return patient_results, test_error, auc, acc_logger, mae_value, r_value
