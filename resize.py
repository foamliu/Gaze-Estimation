import os
from multiprocessing import Pool

import cv2 as cv
from tqdm import tqdm

im_size = 256

new_folder = 'data/cron20190326_resized'
old_folder = 'data/data/frame/cron20190326'


def resize_images(d):
    dir = os.path.join(old_folder, d)
    files = [f for f in os.listdir(dir) if f.endswith('.jpg')]

    dst_folder = os.path.join(new_folder, d)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)

    for f in files:
        img_path = os.path.join(dir, f)
        img = cv.imread(img_path)
        img = cv.resize(img, (im_size, im_size))
        dst_file = os.path.join(dst_folder, f)
        cv.imwrite(dst_file, img)


if __name__ == "__main__":
    if not os.path.isdir(new_folder):
        os.makedirs(new_folder)

    dirs = [d for d in os.listdir(old_folder) if os.path.isdir(os.path.join(old_folder, d))]
    with Pool(6) as p:
        r = list(tqdm(p.imap(resize_images, dirs), total=len(dirs)))
