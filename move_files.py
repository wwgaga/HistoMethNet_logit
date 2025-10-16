import os
import shutil

root_dir = os.getcwd()  # Specify the root directory if different

for subdir, dirs, files in os.walk(root_dir):
    if subdir == root_dir:
        continue
    for file in files:
        src_path = os.path.join(subdir, file)
        dest_path = os.path.join(root_dir, file)
        shutil.move(src_path, dest_path)

# Remove empty subdirectories
for subdir, dirs, files in os.walk(root_dir, topdown=False):
    if subdir == root_dir:
        continue
    if not os.listdir(subdir):
        os.rmdir(subdir)
