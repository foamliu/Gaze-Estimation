import os
import pickle

import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR, im_size, pickle_file

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
        label = sample['label']

        # print(filename)
        img = cv.imread(full_path)  # BGR
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = self.transformer(img)  # RGB

        return img, label

    def __len__(self):
        return len(self.samples)
