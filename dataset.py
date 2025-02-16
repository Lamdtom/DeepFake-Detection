from torch.utils.data import Dataset
from PIL import Image
import cv2
import os

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.categories = ['real', 'fake']
        
        for label, category in enumerate(self.categories):
            category_dir = os.path.join(self.root_dir, category)
            for img_name in os.listdir(category_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(category_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label