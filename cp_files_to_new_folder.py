import pandas as pd
import os
import shutil
from pathlib import Path

# Define paths
source_folders = [
    '/cbica/home/tianyu/dataset/Natalie_Cohort',
    '/cbica/home/tianyu/dataset/New_set_from_Natalie_11_2024',
    '/cbica/home/tianyu/dataset/natalie_jan_new_data',
    '/cbica/home/tianyu/dataset/GBM_NIH_All'
]
destination_folder = '/cbica/home/tianyu/dataset/penn_nih_combine_slides'
csv_file = 'dataset_csv/dna_methylation_upenn_nih_new_jan.csv'
not_found_file = 'unfound_slides.txt'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Initialize lists to track found and not found files
not_found_slides = []
found_slides = []

dest_files = os.listdir(destination_folder)
print(dest_files)

# Process each slide ID
for slide_id in df['slide_id']:
    slide_filename = f"{slide_id}.ndpi"
    found = False
    
    # Search in each source folder
    for folder in source_folders:
        source_path = os.path.join(folder, slide_filename)
        
        if os.path.exists(source_path):
            # Copy file to destination
            dest_path = os.path.join(destination_folder, slide_filename)
            if slide_filename in dest_files:
                print('skipped {} due to already exist'.format(slide_filename))
                found=True
                continue 

            shutil.copy2(source_path, dest_path)
            found = True
            found_slides.append(slide_filename)
            print(f"Copied: {slide_filename}")
            break
    
    if not found:
        not_found_slides.append(slide_filename)
        print(f"Not found: {slide_filename}")

# Write not found slides to file
with open(not_found_file, 'w') as f:
    f.write('\n'.join(not_found_slides))

# Print summary
print(f"\nSummary:")
print(f"Total slides processed: {len(df)}")
print(f"Files found and copied: {len(found_slides)}")
print(f"Files not found: {len(not_found_slides)}")
print(f"List of unfound slides saved to: {not_found_file}")
