import argparse
import math
import os
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import num_tests, IMG_DIR
from data_gen import data_transforms
from utils import ensure_folder

angles_file = 'data/angles.txt'
test_file = 'data/test_pairs_rectified.txt'
IMG_FOLDER = 'data/data/frame/cron20190326'
transformer = data_transforms['val']


def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def get_image(file, flip=False):
    file = os.path.join(IMG_DIR, file)
    img = cv.imread(file)
    # img = cv.resize(img, (im_size, im_size))
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


# def get_feature(model, file):
#     imgs = torch.zeros([2, 3, 224, 224], dtype=torch.float, device=device)
#     imgs[0] = get_image(file, False)
#     imgs[1] = get_image(file, True)
#     with torch.no_grad():
#         output = model(imgs)
#     feature_0 = output[0].cpu().numpy()
#     feature_1 = output[1].cpu().numpy()
#     feature = feature_0 + feature_1
#     return feature / np.linalg.norm(feature)


def get_feature(model, file):
    img = get_image(file)
    imgs = img.unsqueeze(dim=0)
    with torch.no_grad():
        output = model(imgs)
    feature = output[0].cpu().numpy()
    return feature / np.linalg.norm(feature)


def evaluate(model):
    model.eval()

    with open(test_file, 'r') as file:
        lines = file.readlines()

    angles = []

    elapsed = 0.0

    for line in tqdm(lines):
        tokens = line.split()

        start = time.time()
        x0 = get_feature(model, tokens[0])
        x1 = get_feature(model, tokens[1])
        end = time.time()
        elapsed += (end - start)

        cosine = np.dot(x0, x1)
        cosine = np.clip(cosine, -1, 1)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi
        is_same = tokens[2]
        angles.append('{} {} {} {} \n'.format(theta, is_same, tokens[0], tokens[1]))

    elapsed_time = elapsed / (num_tests * 2) * 1000
    print('elapsed time per image: {} ms'.format(elapsed_time))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


def visualize(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    bins = np.linspace(0, 180, 181)

    plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
    plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))
    print('mu_0: ' + str(mu_0))
    print('sigma_0: ' + str(sigma_0))
    print('mu_1: ' + str(mu_1))
    print('sigma_1: ' + str(sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    ensure_folder('images')
    plt.savefig('images/theta_dist.png')
    plt.show()


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1 and angle > threshold or type == 0 and angle <= threshold:
            wrong += 1

    accuracy = 1 - wrong / num_tests
    return accuracy


def error_analysis(threshold):
    with open(angles_file) as file:
        angle_lines = file.readlines()

    fp = []
    fn = []
    fp_lines = []
    fn_lines = []
    for i, line in enumerate(angle_lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        file0 = tokens[2]
        file1 = tokens[3]
        new_line = '{} {} {}\n'.format(file0, file1, type)
        if angle <= threshold and type == 0:
            fp.append(i)
            fp_lines.append(new_line)
        if angle > threshold and type == 1:
            fn.append(i)
            fn_lines.append(new_line)

    print('len(fp): ' + str(len(fp)))
    print('len(fn): ' + str(len(fn)))

    num_fp = len(fp)
    num_fn = len(fn)

    filename = 'data/test_pairs.txt'
    with open(filename, 'r') as file:
        pair_lines = file.readlines()

    for i in range(num_fp):
        fp_id = fp[i]
        fp_line = pair_lines[fp_id]
        tokens = fp_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fp_0.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fp_1.jpg'.format(i))

    for i in range(num_fn):
        fn_id = fn[i]
        fn_line = pair_lines[fn_id]
        tokens = fn_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fn_0.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fn_1.jpg'.format(i))

    with open('data/errors.txt', 'w') as file:
        file.write('FP:\n')
        file.writelines(fp_lines)
        file.write('FN:\n')
        file.writelines(fn_lines)


def copy_file(old, new):
    old = os.path.join(IMG_DIR, old)
    # print(old)
    img = cv.imread(old)
    new_fn = os.path.join('images', new)
    cv.imwrite(new_fn, img)


def get_threshold():
    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 6000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def test(model):
    print('Evaluating {}...'.format(angles_file))
    evaluate(model)

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold()
    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
    return acc, thres


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # if not args.gpu:
    device = torch.device('cpu')
    # else:
    #    from config import device
    print('test with {}'.format(device))

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model = checkpoint['model'].module

    # filename = 'image_matching_mobile.pt'
    # model = MobileNetV2()
    # model.load_state_dict(torch.load(filename))

    # class HParams:
    #     def __init__(self):
    #         self.pretrained = False
    #         self.use_se = True
    # filename = 'image-matching.pt'
    # from models import resnet50
    # model = resnet50(HParams())
    # model.load_state_dict(torch.load(filename))

    # scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
    # model = torch.jit.load(scripted_quantized_model_file)

    # scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
    # model = torch.jit.load(scripted_float_model_file)

    model = model.to(device)
    model.eval()
    # print(model)

    acc, threshold = test(model)

    print('Visualizing {}...'.format(angles_file))
    visualize(threshold)

    print('error analysis...')
    error_analysis(threshold)
