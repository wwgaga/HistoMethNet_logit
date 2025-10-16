import os

def list_files_to_txt(directory, output_file="files_names.txt"):
    """Gets a list of files in a directory and writes them to a text file."""

    with open(output_file, "w") as f:
        for file_name in os.listdir(directory):
            f.write(file_name + "\n")

if __name__ == "__main__":
    directory_path = "/cbica/home/tianyu/dataset/NIH_features_extracted/pt_files"  # Replace with your directory path
    list_files_to_txt(directory_path)