import cv2
import numpy as np
from PIL import Image, ImageOps
import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm, rcParams
from matplotlib import patches as mpatches
import seaborn as sns

from PIL import Image
# from mil_models import create_model
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

import sys
sys.path.append('../')
from models.rrt import RRTMIL
import torch

def get_mixture_plot(mixtures, cell_type_names):
    # colors = [
    #     '#696969','#556b2f','#a0522d','#483d8b', 
    #     '#008000','#008b8b','#000080','#7f007f',
    #     '#8fbc8f','#b03060','#ff0000','#ffa500',
    #     '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
    #     '#00ffff','#00bfff','#f4a460','#adff2f',
    #     '#da70d6','#b0c4de','#ff00ff','#1e90ff',
    #     '#f0e68c','#0000ff','#dc143c','#90ee90',
    #     '#ff1493','#7b68ee','#ffefd5','#ffb6c1']
    colors = [
        '#00bfff','#696969','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']
   
    cmap = {f'{cell_type_names[k]}':v for k,v in enumerate(colors[:len(mixtures)])}
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(12,6), dpi=600)  # Increased figure size

    prop = fm.FontProperties(fname="./Arial.ttf", weight='bold')  # Set font to bold
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mixtures = pd.DataFrame(mixtures, index=cmap.keys()).T
    ax = sns.barplot(mixtures, palette=cmap)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_xlabel('Cell Type', fontproperties=prop, fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion', fontproperties=prop, fontsize=14, fontweight='bold')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontproperties=prop, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45, ha='right', fontproperties=prop, fontsize=12, fontweight='bold')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    # Calculate percentages of each mixture without using 'keepdims'
    mixtures_sum = mixtures.sum(axis=1)  # Removed keepdims=True
    percentages = mixtures.div(mixtures_sum, axis=0) * 100

    # Optionally, annotate the bar plot with percentage values
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{percentages.iloc[0, i]:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)
    
    # Make axis labels and title bold
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')
    ax.title.set_weight('bold')

    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.close()
    return ax.get_figure(), percentages  # Modified to return percentages


def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    # colors = [
    #     '#696969','#556b2f','#a0522d','#483d8b', 
    #     '#008000','#008b8b','#000080','#7f007f',
    #     '#8fbc8f','#b03060','#ff0000','#ffa500',
    #     '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
    #     '#00ffff','#00bfff','#f4a460','#adff2f',
    #     '#da70d6','#b0c4de','#ff00ff','#1e90ff',
    #     '#f0e68c','#0000ff','#dc143c','#90ee90',
    #     '#ff1493','#7b68ee','#ffefd5','#ffb6c1']
    
    colors = [
        '#00bfff', '#696969','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f', '#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']

    colors = colors[:n]
    # colors = colors[:n] 
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict

def get_rrt_encoder(ckpt_path, device='cuda'):
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
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
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_mt', 'rrt'], default='clam_sb', 
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
    parser.add_argument('--cell_property', default='cell_type', type=str, help='cell property choice for propotion loss: cell_type | stem | immune')

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
    args = parser.parse_args()
    
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
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'cell_property': args.cell_property,
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

def visualize_categorical_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        canvas_color=(255, 255, 255),
        alpha=0.4,
        verbose=True,
    ):

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size  

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        # Fix KeyError by converting tensor to int
        color = label2color_dict[int(labels[idx][0].item())]  # Convert tensor to int
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img




