import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader



BATCH_SIZE = 2
n_buildings = 1867
n_no_buildings = 20134
weights = [1/n_buildings, 1/n_no_buildings]
sampler = WeightedRandomSampler(weights=weights, num_samples=BATCH_SIZE)

class SatelliteClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.no_build_ids = sorted(os.listdir(os.path.join(data_dir, 'no_buildings')))
        self.build_ids = sorted(os.listdir(os.path.join(data_dir, 'buildings')))
        img_label_tuple = [(i, 1) for i in self.build_ids] + [(i, 0) for i in self.no_build_ids]
        self.ids = sorted(img_label_tuple)
        self.transform = transform

    def __getitem__(self, index):
        id, label = self.ids[index]
        if label == 1:
            image_path = os.path.join(self.data_dir, 'buildings', id)
        else:
            image_path = os.path.join(self.data_dir, 'no_buildings', id)
        image = Image.open(image_path)
        image.load()
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.ids)



class SatelliteSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.ids = sorted(os.listdir(os.path.join(data_dir, 'images')))
        self.masks = sorted(os.listdir(os.path.join(data_dir, 'images')))
        self.transform = transform

    def __getitem__(self, index):
        id = self.ids[index]
        image = Image.open(os.path.join(self.data_dir, 'images', id))
        mask = Image.open(os.path.join(self.data_dir, 'masks', id))
        image.load()
        mask.load()
        if self.transform:
            image, mask = self.transform((image, mask))
        return image, mask

    def __len__(self):
        return len(self.ids)
