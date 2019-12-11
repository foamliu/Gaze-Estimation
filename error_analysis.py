import os

import cv2 as cv

from utils import draw_str

IMG_FOLDER = 'data/data/frame/cron20190326'
im_size = 448


def do_image(src, dst):
    title = src
    src = os.path.join(IMG_FOLDER, src)
    dst = os.path.join('images', dst)

    img = cv.imread(src)
    img = cv.resize(img, (im_size, im_size))
    draw_str(img, (20, 20), title)
    cv.imwrite(dst, img)


if __name__ == "__main__":

    with open('data/errors.txt', 'r') as file:
        lines = file.readlines()

    fp_lines = []
    fn_lines = []
    curr = 'fp'
    i = 0
    for line in lines:
        if line.startswith('FP:'):
            curr = 'fp'
            i = 0
            continue
        if line.startswith('FN:'):
            curr = 'fn'
            i = 0
            continue

        tokens = line.split()
        do_image(tokens[0], '{}_{}_{}.jpg'.format(i, curr, 0))
        do_image(tokens[1], '{}_{}_{}.jpg'.format(i, curr, 1))
        i += 1
