import json
import os
import pickle
import random

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from config import device, im_size, IMG_DIR
from data_gen import data_transforms
from utils import ensure_folder


def save_images(full_path, filename, i):
    raw = cv.imread(full_path)
    resized = cv.resize(raw, (im_size, im_size))
    cv.imwrite('images/{}_raw.jpg'.format(i), resized)

    img = cv.imread(os.path.join(IMG_DIR, filename))
    img = cv.resize(img, (im_size, im_size))
    cv.imwrite('images/{}_img.jpg'.format(i), img)


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open('data/val.pkl', 'rb') as fp:
        data = pickle.load(fp)

    samples = data
    samples = random.sample(samples, 10)

    inputs = torch.zeros([10, 3, im_size, im_size], dtype=torch.float, device=device)

    transformer = data_transforms['val']

    sample_preds = []
    ensure_folder('images')

    for i, sample in enumerate(samples):
        filename = sample['filename']
        full_path = os.path.join(IMG_DIR, filename)
        print(full_path)
        save_images(full_path, filename, i)

        img = Image.open(full_path)
        img = transformer(img)
        inputs[i] = img
        label = sample['label']
        label = [np.rad2deg(l) for l in label]
        sample_preds.append({'filename': filename, 'label': label})

    with torch.no_grad():
        out = model(inputs)

    print('out.size(): ' + str(out.size()))
    out = out.cpu().numpy()
    print('out: ' + str(out))

    for i in range(10):
        sample = sample_preds[i]
        ret = out[i].tolist()
        ret = [np.rad2deg(l) for l in ret]
        sample['out'] = ret

    with open('sample_preds.json', 'w') as file:
        json.dump(sample_preds, file, indent=4, ensure_ascii=False)
