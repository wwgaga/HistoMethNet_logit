import h5py
import openslide
import torch
import pandas as pd
import argparse
import os

from prototype_visualization_utils import visualize_categorical_heatmap, get_mixture_plot, get_default_cmap
# from utils.eval_utils import initiate_model as initiate_model
# from models import get_encoder
import sys
from prototype_visualization_utils import get_rrt_encoder
sys.path.append('../')
# from mil_models.tokenizer import PrototypeTokenizer

print('\ninitializing model from checkpoint')
# ckpt_path = '/cbica/home/tianyu/projects/dna-rrt/results/task_dna_subtyping_RRT_s1/s_6_checkpoint_best.pt'
# ckpt_path = '/cbica/home/tianyu/projects/dna-rrt/results/task_dna_subtyping_RRT_Mixbag_stem_best_auc_s1/s_6_checkpoint_best.pt'
ckpt_path = '/cbica/home/tianyu/projects/dna-rrt/results/task_dna_subtyping_RRT_Mixbag_immune_s1/s_6_checkpoint_best.pt'

print('\nckpt path: {}'.format(ckpt_path))
device = 'cuda:0'
feature_extractor = get_rrt_encoder(ckpt_path, device=device)

print('Done!')

# Read slide IDs from CSV
csv_path = '../dataset_csv/dna_methylation_upenn.csv'  # Update this path as needed
df = pd.read_csv(csv_path)
slide_ids = df['slide_id'].tolist()

# cell_type_names = ['endothelial', 'GN', 'lymphoid_cells', 'MES_ATYP', 'MES_TYP', 'myeloid_cells', 'RTK_I', 'RTK_II']
cell_type_names = ['immune', 'non-immune']

for slide_id in slide_ids:
    print(f"\nProcessing slide: {slide_id}")
    
    slide_fpath = f'/cbica/home/tianyu/dataset/Natalie_Cohort/{slide_id}.ndpi'
    h5_feats_fpath = f'/cbica/home/tianyu/dataset/Natalie_Cohort_UNI_Features/h5_files/{slide_id}.h5'
    
    if not os.path.exists(slide_fpath) or not os.path.exists(h5_feats_fpath):
        print(f"Skipping {slide_id}: Files not found")
        continue

    wsi = openslide.open_slide(slide_fpath)
    h5 = h5py.File(h5_feats_fpath, 'r')

    coords = h5['coords'][:]
    feats = torch.Tensor(h5['features'][:])
    patch_size = 256

    ### get PANTHER representation and GMM mixtures
    with torch.inference_mode():
        feats = feats.to(device)
        logits, instance_dict = feature_extractor(feats)
        cell_type_patch_level = instance_dict['cell_type_logits']
        cell_type_prob = instance_dict['cell_type_prob']
        
        global_cluster_labels = cell_type_patch_level.argmax(axis=1)
    
    ### Visualize the categorical heatmap and the GMM mixtures
    cat_map = visualize_categorical_heatmap(
        wsi,
        coords, 
        global_cluster_labels, 
        label2color_dict=get_default_cmap(2),
        vis_level=wsi.get_best_level_for_downsample(64), # original 128
        patch_size=(patch_size, patch_size),
        alpha=0.4,
    )

    # Get mixture plot and percentages
    mus, percentages = get_mixture_plot(cell_type_prob.detach().cpu().numpy(), cell_type_names)

    # Save the produced figure
    mus.savefig(f'immune/precentage_maps/percentages_{slide_id}.jpg', format='JPEG')

    # Store the percentages in a CSV file
    pd.DataFrame(percentages, columns=[f'{cell_type_names[i]}' for i in range(percentages.shape[1])]).to_csv(f'immune/precentage_csv/cell_type_percentages_{slide_id}.csv', index=False)

    # Save the heatmap
    cat_map.save(f'immune/categorical_maps/cell_type_heatmap_{slide_id}.jpg', 'JPEG')

    # Close open files
    h5.close()
    wsi.close()

print("Processing complete!")
