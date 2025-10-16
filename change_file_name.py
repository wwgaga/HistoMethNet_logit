import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if '_P_' in filename:
            new_filename = filename.replace('_P_', '_')
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

if __name__ == "__main__":
    folder_path = '/cbica/home/tianyu/dataset/Natalie_Cohort'  # Update this path
    rename_files_in_folder(folder_path)



