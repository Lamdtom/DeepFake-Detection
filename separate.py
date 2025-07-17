import os
import random
import shutil
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

def create_split_directories(output_dir):
    # Define the new directories for train, validation, and test splits
    splits = ['train', 'val', 'test']
    categories = ['real', 'fake']
    
    for split in splits:
        for category in categories:
            split_path = os.path.join(output_dir, split, category)
            os.makedirs(split_path, exist_ok=True)

def split_dataset(data_dir, output_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    # Check if train, val, and test directories exist, if not create them
    create_split_directories(output_dir)
    
    categories = ['real', 'fake']

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        files = os.listdir(category_dir)

        # Shuffle the files
        random.shuffle(files)

        # Compute the number of files for each split
        total_files = len(files)
        train_count = int(total_files * train_size)
        val_count = int(total_files * val_size)
        test_count = total_files - train_count - val_count

        # Split the files
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # Move the files to the respective directories
        for split, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            for file in file_list:
                source = os.path.join(category_dir, file)
                destination = os.path.join(output_dir, split, category, file)
                shutil.copy(source, destination)

# Set paths
data_dir = 'deepfake_dataset'  # Path to your dataset (with real and fake folders)
output_dir = 'dataset'  # Path to where the train, val, test directories will be created

# Split the dataset into train, validation, and test
split_dataset(data_dir, output_dir)
