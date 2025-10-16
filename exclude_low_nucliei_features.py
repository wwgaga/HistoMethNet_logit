import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from histolab.scorer import NucleiScorer
from histolab.tile import Tile

# Set device for torch operations
device = torch.device('cuda' if torch.cuda.is_available() else 
                     ('mps' if torch.backends.mps.is_available() else 'cpu'))

print(f"Using device: {device}")

# Initialize nuclei scorer once to reuse
nuclei_scorer = NucleiScorer()

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches')

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def compute_nuclei_score(image):
    """
    Compute nuclei score for a given patch image using histolab's NucleiScorer.
    
    Args:
        image: PIL Image object of the patch
        
    Returns:
        float: Raw nuclei score (higher value = more nuclei)
    """
    try:
        # Make sure image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            # Convert numpy array to PIL Image if needed
            image = Image.fromarray(image).convert('RGB')
        
        # Create a Tile object that histolab expects
        tile = Tile(image, coords=(0, 0), level=0)
        
        # Get raw nuclei score from histolab's scorer
        raw_score = nuclei_scorer(tile)
        
        return float(raw_score)
        
    except Exception as e:
        print(f"Error computing nuclei score: {e}")
        return 0.0

def process_slide_features(h5_file_path, slide_file_path, nuclei_threshold=0.03, patch_size=224):
    """
    Process features from h5 file and filter based on nuclei scores.
    
    Args:
        h5_file_path: Path to the h5 file containing features and coordinates
        slide_file_path: Path to the corresponding slide file
        nuclei_threshold: Threshold for nuclei score filtering
        patch_size: Size of patches to extract (default 224 for UNI model)
        
    Returns:
        tuple: (filtered_features, filtered_coords, nuclei_scores)
    """
    print(f"Processing {h5_file_path}")
    print(f"Corresponding slide: {slide_file_path}")
    
    # Try different slide extensions if the exact path doesn't exist
    if not os.path.exists(slide_file_path):
        slide_base = os.path.splitext(slide_file_path)[0]
        possible_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs']
        for ext in possible_extensions:
            if os.path.exists(slide_base + ext):
                slide_file_path = slide_base + ext
                print(f"Found slide with extension {ext}: {slide_file_path}")
                break
    
    if not os.path.exists(slide_file_path):
        print(f"WARNING: Slide not found: {slide_file_path}")
        return None, None, None
    
    # Open the H5 file containing features and coordinates
    with h5py.File(h5_file_path, "r") as file:
        if 'features' not in file or 'coords' not in file:
            print(f"ERROR: H5 file {h5_file_path} missing features or coords datasets")
            return None, None, None
        
        features = file['features'][:]
        coords = file['coords'][:]
    
    print(f"Loaded {len(features)} features")
    
    try:
        # Open the slide
        wsi = openslide.OpenSlide(slide_file_path)
        
        nuclei_scores = []
        valid_indices = []
        
        # Process each patch
        for idx, coord in enumerate(tqdm(coords, desc="Scoring patches")):
            try:
                # Extract patch from slide - coordinates are x,y at level 0
                x, y = coord
                patch = wsi.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
                
                # Compute nuclei score
                score = compute_nuclei_score(patch)
                nuclei_scores.append(score)
                
                # Keep patches with high nuclei scores
                if score >= nuclei_threshold:
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing patch at coordinates {coord}: {e}")
                nuclei_scores.append(0)  # Assign zero score to problematic patches
        
        # Filter features based on nuclei scores
        if valid_indices:
            filtered_features = features[valid_indices]
            filtered_coords = coords[valid_indices]
            
            print(f"Retained {len(filtered_features)}/{len(features)} patches ({len(filtered_features)/len(features)*100:.1f}%)")
            
            return filtered_features, filtered_coords, np.array(nuclei_scores)
        else:
            print("WARNING: No patches passed the nuclei threshold")
            return np.array([]), np.array([]), np.array(nuclei_scores)
        
    except Exception as e:
        print(f"ERROR processing slide {slide_file_path}: {e}")
        return None, None, None

def main():
	parser = argparse.ArgumentParser(description='Feature Filtering based on Nuclei Scores')
	parser.add_argument('--feature_dir', type=str, default='/Volumes/Yu_SSD_2TB/TCGA_new_features_extracted/h5_files',
						help='Directory containing extracted feature h5 files')
	parser.add_argument('--slide_dir', type=str, default='/Volumes/Yu_SSD_2TB/TCGA_selected_slides',
						help='Directory containing whole slide images')
	parser.add_argument('--output_dir', type=str, default='filtered_features',
						help='Directory to save filtered features')
	parser.add_argument('--nuclei_threshold', type=float, default=0.025,
						help='Threshold for nuclei score (higher means more nuclei)')
	parser.add_argument('--patch_size', type=int, default=256,
						help='Size of patches in pixels')
	parser.add_argument('--slide_extension', type=str, default='.ndpi',
						help='File extension for slide files')
	args = parser.parse_args()

	print('Initializing nuclei-based feature filtering')
	print(f'Feature directory: {args.feature_dir}')
	print(f'Slide directory: {args.slide_dir}')
	print(f'Output directory: {args.output_dir}')
	print(f'Nuclei threshold: {args.nuclei_threshold}')
	
	# Create output directories
	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, 'h5_files'), exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, 'pt_files'), exist_ok=True)

	# Process each H5 file
	h5_files = [f for f in os.listdir(args.feature_dir) if f.endswith('.h5')]
	print(f'Found {len(h5_files)} h5 files to process')
	
	# Summary statistics
	total_patches = 0
	retained_patches = 0
	processed_slides = 0
	slides_with_errors = 0
	
	for h5_file in tqdm(h5_files, desc="Processing slides"):
		try:
			slide_id = h5_file.split('.h5')[0]
			slide_path = os.path.join(args.slide_dir, f"{slide_id}{args.slide_extension}")
			h5_path = os.path.join(args.feature_dir, h5_file)
			# Save filtered features
			output_h5_path = os.path.join(args.output_dir, 'h5_files', h5_file)
			print(f"\n{'='*50}")
			print(f"Processing {slide_id}")
			
			if not os.path.exists(slide_path):
				print('skipped {} due to missing slide file'.format(slide_id))
				continue 		
			
			if os.path.exists(output_h5_path):
				print('skipped {} due to have h5 file already generate'.format(slide_id))
				continue 	
				
			# Process features and get nuclei scores
			filtered_features, filtered_coords, nuclei_scores = process_slide_features(
				h5_path, slide_path, args.nuclei_threshold, args.patch_size
			)
			
		

			if filtered_features is None:
				# Error occurred during processing
				slides_with_errors += 1
				continue
				
			if len(nuclei_scores) == 0:
				print(f"No patches found in {slide_id}")
				continue
				
			# Update statistics
			processed_slides += 1
			total_patches += len(nuclei_scores)
			retained_patches += len(filtered_features)
			
			
			try:
				with h5py.File(output_h5_path, 'w') as file:
					if len(filtered_features) > 0:
						file.create_dataset('features', data=filtered_features)
						file.create_dataset('coords', data=filtered_coords)
					file.create_dataset('nuclei_scores', data=nuclei_scores)
					
				# Save as PT file if there are filtered features
				if len(filtered_features) > 0:
					output_pt_path = os.path.join(args.output_dir, 'pt_files', f"{slide_id}.pt")
					torch.save(torch.from_numpy(filtered_features), output_pt_path)
				
				print(f"Original patches: {len(nuclei_scores)}")
				print(f"Filtered patches: {len(filtered_features)}")
				print(f"Average nuclei score: {np.mean(nuclei_scores):.3f}")
				print(f"Filtered data saved to {output_h5_path}")
			except Exception as e:
				print(f"Error saving filtered data for {slide_id}: {e}")
				slides_with_errors += 1
				
		except Exception as e:
			print(f"Error processing {h5_file}: {e}")
			slides_with_errors += 1
	
	# Final summary
	print("\n" + "="*50)
	print("FILTERING COMPLETE")
	print(f"Processed {processed_slides} slides")
	print(f"Encountered errors in {slides_with_errors} slides")
	if total_patches > 0:
		print(f"Retained {retained_patches}/{total_patches} patches ({retained_patches/total_patches*100:.1f}%)")
	print(f"Results saved to {args.output_dir}")
	print("="*50)

if __name__ == '__main__':
	main()


