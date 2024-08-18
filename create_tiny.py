import os
import shutil
import random

# Specify the root directory and the subdirectories you need
root_dir = '/depot/yunglu/data/datasets_ben/maestro/maestro_with_mistakes_unaligned'
subdirs = ['mistake', 'score', 'label/correct_notes', 'label/removed_notes', 'label/extra_notes']
max_size = 12 * 1024 * 1024  # 12MB in bytes
output_dir = '/depot/yunglu/data/datasets_ben/temp_MAESTRO_v3'  # Specify the directory where you want to save the selected files

# Initialize total size
total_size = 0

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to get the size of a file or directory
def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total
    return 0

# Get all unique numbers across all subdirectories
all_numbers = set()
for subdir in subdirs:
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.exists(subdir_path):
        all_numbers.update(os.listdir(subdir_path))

# Convert to a list and shuffle to randomize the selection order
all_numbers = list(all_numbers)
random.shuffle(all_numbers)

# Iterate through each number and include all corresponding folders from all subdirectories
for number in all_numbers:
    if total_size >= max_size:
        break

    for subdir in subdirs:
        folder_path = os.path.join(root_dir, subdir, number)
        if os.path.exists(folder_path):
            folder_size = get_size(folder_path)

            # Check if adding this folder exceeds the max size
            if total_size + folder_size > max_size:
                print(f"Reached the size limit of {max_size / (1024 * 1024)}MB. Stopping selection.")
                break

            # Copy the folder to the output directory, preserving the structure
            destination_subdir = os.path.join(output_dir, subdir, number)

            if os.path.isdir(folder_path):
                # If the destination directory exists, remove it
                if os.path.exists(destination_subdir):
                    shutil.rmtree(destination_subdir)
                shutil.copytree(folder_path, destination_subdir)
            else:
                os.makedirs(destination_subdir, exist_ok=True)
                shutil.copy2(folder_path, destination_subdir)

            # Update total size
            total_size += folder_size

        if total_size >= max_size:
            break

print(f"Total selected size: {total_size / (1024 * 1024):.2f} MB")
print(f"Files and directories are stored in: {output_dir}")