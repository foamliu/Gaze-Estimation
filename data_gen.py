import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class GazeEstimationDataset(Dataset):
    def __init__(self, split):
        with open('data/{}.pkl'.format(split), 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['filename']
        full_path = os.path.join(IMG_DIR, filename)
        look_vec = sample['look_vec']
        pupil_size = sample['look_vec']
        label = np.array((4,), dtype=np.float)
        print(look_vec)
        print(type(look_vec))
        label[:3] = list(look_vec)
        label[3] = pupil_size

        # print(filename)
        img = Image.open(full_path)
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)
