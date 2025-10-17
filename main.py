from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_MIL_Dataset_cell_type

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# Add wandb import
import wandb

def main(args):
    # Initialize wandb at the beginning of main
    if args.log_data:
        wandb.init(project=f"Experiment_{args.exp_code}_Seed_{args.seed}", config=args)

        
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_r2_stem = []
    all_r2_immune = []
    all_r2_cell_type = []
    all_mae_cell_type = []
    folds = np.arange(start, end)
    
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
        )
        
        
        datasets = (train_dataset, val_dataset, test_dataset)
        
        results, test_auc, val_auc, test_acc, \
            val_acc, r_value, mae_value = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # all_r2_stem.append(r_value)
        # all_r2_immune.append(mean_r2_score_immune)
        all_mae_cell_type.append(mae_value)
        all_r2_cell_type.append(r_value)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc, 'pearson_r_cell_type':all_r2_cell_type, 'mae_cell_type':all_mae_cell_type })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    # # Save settings to wandb
    # if args.log_data:
    #     wandb.config.update(args)
    
    # with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    #     print(settings, file=f)
    #     # Log settings to wandb
    #     if args.log_data:
    #         wandb.config.update(settings)
    # f.close()
    
    if args.log_data:
        wandb.finish()
    
    return results

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1536)
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
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_mt', 'rrt', 'rrt_logit', 'logit_only'], default='clam_sb', 
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
parser.add_argument('--input_dim', default=1536, type=int, help='Dimension of input features. PLIP features should be [512]')
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
parser.add_argument('--region_num', default=64, type=int, help='Number of regions [8,12,16,...]')
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
parser.add_argument('--cell_property', default='cell_type', type=str, help='cell property choice for propotion loss: cell_type | stem | immune')


# CR-MSA specific arguments
parser.add_argument('--cr_msa', action='store_false', help='Disable CR-MSA')
parser.add_argument('--crmsa_k', default=3, type=int, help='Kernel size for CR-MSA [1,3,5]')
parser.add_argument('--crmsa_heads', default=8, type=int, help='Number of heads in CR-MSA [1,8,...]')
parser.add_argument('--crmsa_mlp', action='store_true', help='Enable MLP phi in CR-MSA')

# DAttention specific arguments
parser.add_argument('--da_act', default='relu', type=str, help='Activation function in DAttention [gelu, relu]')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "mps")



def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/dna_methylation_upenn_debug.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'RTK1':0, 'RTK2':1, 'mesenchymal':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb', 'clam_mt', 'rrt', 'rrt_logit', 'logit_only']:
        assert args.subtyping 

elif args.task == 'task_3_cell_type_classification':
    args.n_classes=3
    
    dataset = Generic_MIL_Dataset_cell_type(csv_path = 'dataset_csv/dna_methylation_upenn_nih_new_jan.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'RTK1':0, 'RTK2':1, 'mesenchymal':2},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['clam_sb', 'clam_mb', 'clam_mt', 'rrt', 'rrt_logit', 'logit_only']:
        assert args.subtyping 
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")

