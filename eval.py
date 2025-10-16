from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
# from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_MIL_Dataset_cell_type

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
# parser.add_argument('--results_dir', type=str, default='./results',
                    # help='relative path to results folder, i.e. '+
                    # 'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
# parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
#                     help='size of model (default: small)')
# parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'rrt'], default='clam_sb', 
                    # help='type of model (default: clam_sb)')
# parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
# parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
# parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')

parser.add_argument('--cell_property', default='stem', type=str, help='cell property choice for propotion loss: cell_type | stem | immune')

parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                    help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_mt', 'rrt', 'rrt_logit'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='big', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_3_cell_type_classification'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                    help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                    help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                    help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

# Model-specific arguments for RRT
parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
parser.add_argument('--aux_alpha', default=1.0, type=float, help='Auxiliary loss alpha')
parser.add_argument('--input_dim', default=1024, type=int, help='Dimension of input features. PLIP features should be [512]')
parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention mechanism')
parser.add_argument('--pool', default='attn', type=str, help='Classification pooling method. Use abMIL.')
parser.add_argument('--ffn', action='store_true', help='Enable Feed-Forward Network (only for ablation studies)')
parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of Transformer layers')
parser.add_argument('--mlp_ratio', default=4.0, type=float, help='MLP ratio in the Feed-Forward Network')
parser.add_argument('--qkv_bias', action='store_false', help='Disable bias in QKV projections')
parser.add_argument('--all_shortcut', action='store_true', help='Enable shortcut connections x = x + rrt(x)')

# R-MSA specific arguments
parser.add_argument('--region_attn', default='native', type=str, help='Region attention type (only for ablation)')
parser.add_argument('--min_region_num', default=0, type=int, help='Minimum number of regions (only for ablation)')
parser.add_argument('--region_num', default=32, type=int, help='Number of regions [8,12,16,...]')
parser.add_argument('--trans_dim', default=64, type=int, help='Transformer dimension (only for ablation)')
parser.add_argument('--n_heads', default=8, type=int, help='Number of heads in R-MSA')
parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout rate in R-MSA')
parser.add_argument('--drop_path', default=0.0, type=float, help='Drop path rate in R-MSA')

# PEG or PPEG specific arguments
parser.add_argument('--pos', default='none', type=str, help='Position embedding type, enable PEG or PPEG')
parser.add_argument('--pos_pos', default=0, type=int, help='Position of positional embedding [-1,0]')
parser.add_argument('--peg_k', default=7, type=int, help='Kernel size for PEG and PPEG')
parser.add_argument('--peg_1d', action='store_true', help='Enable 1-D PEG and PPEG')

# EPEG specific arguments
parser.add_argument('--epeg', action='store_false', help='Disable EPEG')
parser.add_argument('--epeg_bias', action='store_false', help='Disable convolution bias in EPEG')
parser.add_argument('--epeg_2d', action='store_true', help='Enable 2D convolution in EPEG (only for ablation)')
parser.add_argument('--epeg_k', default=15, type=int, help='Kernel size for EPEG [9,15,21,...]')
parser.add_argument('--epeg_type', default='attn', type=str, help='Type of EPEG (only for ablation)')

# CR-MSA specific arguments
parser.add_argument('--cr_msa', action='store_false', help='Disable CR-MSA')
parser.add_argument('--crmsa_k', default=3, type=int, help='Kernel size for CR-MSA [1,3,5]')
parser.add_argument('--crmsa_heads', default=8, type=int, help='Number of heads in CR-MSA [1,8,...]')
parser.add_argument('--crmsa_mlp', action='store_true', help='Enable MLP phi in CR-MSA')
# DAttention specific arguments
parser.add_argument('--da_act', default='tanh', type=str, help='Activation function in DAttention [gelu, relu]')
parser.add_argument('--n_classes', default=3, type=int, help='Number of classes')
parser.add_argument('--data_csv', type=str, default=None, help='data csv file')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir


assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/dna_methylation_upenn_dana_suggest.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/dna_methylation_upenn_dana_suggest.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'task_3_cell_type_classification':
    args.n_classes=3
    
    dataset = Generic_MIL_Dataset_cell_type(csv_path = args.data_csv,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'RTK1':0, 'RTK2':1, 'mesenchymal':2},
                            patient_strat= False,
                            ignore=[])

   
# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint_best_r.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_r_scores = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            # csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            csv_path = '{}/splits_6.csv'.format(args.splits_dir, folds[ckpt_idx])

            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
           
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, val_error, val_auc,mae_value, r_value  = eval(split_dataset, args, ckpt_paths[ckpt_idx], device)
        all_results.append(all_results)
        all_auc.append(val_auc)
        all_acc.append(1-val_error)
        all_r_scores.append(r_value)
        # df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'r_score': all_r_scores})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
