import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []  # Store tuples of (start image, end image) paths
        self.time_skips = []  # Store time skips

        # List all subdirectories in the root_dir
        subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # Iterate through subdirectories to collect image paths and time skips
        for subdir in subdirectories:
            subdir_path = os.path.join(root_dir, subdir)
            image_files = sorted([f for f in os.listdir(os.path.join(subdir_path, "images")) if f.endswith(".tif")])

            for i in range(len(image_files) - 1):
                self.image_paths.append((os.path.join(subdir_path, "images", image_files[i]), os.path.join(subdir_path, "images", image_files[i + 1])))
                time_skip = i + 1
                self.time_skips.append(time_skip)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        start_img_path, end_img_path = self.image_paths[idx]
        time_skip = self.time_skips[idx]

        start_img = Image.open(start_img_path)
        end_img = Image.open(end_img_path)

        if self.transform:
            start_img = self.transform(start_img)
            end_img = self.transform(end_img)

        return start_img, end_img, time_skip