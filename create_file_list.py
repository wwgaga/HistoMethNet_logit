import os

# Specify the directory (current directory in this case)
directory = "/cbica/home/tianyu/dataset/Penn_NIH_Combine_Features/pt_files/"

# Get all entries in the directory
entries = os.listdir(directory)

# Filter out directories, keep only files
filenames = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]

# Write the filenames to a text file
with open('Jan_new_pathology_file_list.txt', 'w') as txt_file:
    for filename in filenames:
        txt_file.write(filename + '\n')

print(f"Successfully written {len(filenames)} pathology_file_list to 'filenames.txt'")