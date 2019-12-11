import os
import pickle
import random

from tqdm import tqdm

from config import IMG_DIR_ALIGNED, pickle_file, num_tests


def get_data():
    samples = []
    dirs = [d for d in os.listdir(IMG_DIR_ALIGNED) if os.path.isdir(os.path.join(IMG_DIR_ALIGNED, d))]
    for d in tqdm(dirs):
        build_vocab(d)

        dir = os.path.join(IMG_DIR_ALIGNED, d)
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]

        for f in files:
            img_path = os.path.join(d, f)
            img_path = img_path.replace('\\', '/')
            samples.append({'img': img_path, 'label': VOCAB[d]})

    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


def pick_one_file(folder):
    files = [f for f in os.listdir(os.path.join(IMG_DIR_ALIGNED, folder)) if f.endswith('.jpg') and not f.endswith('0.jpg')]
    file = random.choice(files)
    file = os.path.join(folder, file)
    file = file.replace('\\', '/')
    return file


if __name__ == "__main__":
    VOCAB = {}
    IVOCAB = {}

    num_same = int(num_tests / 2)
    num_not_same = num_tests - num_same

    out_lines = []
    exclude_list = set()

    picked = set()
    for _ in tqdm(range(num_same)):
        dirs = [d for d in os.listdir(IMG_DIR_ALIGNED) if os.path.isdir(os.path.join(IMG_DIR_ALIGNED, d))]
        folder = random.choice(dirs)
        while len([f for f in os.listdir(os.path.join(IMG_DIR_ALIGNED, folder)) if
                   f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
            folder = random.choice(dirs)

        files = [f for f in os.listdir(os.path.join(IMG_DIR_ALIGNED, folder)) if f.endswith('.jpg') and not f.endswith('0.jpg')]
        file_1 = random.choice(files)
        file_0 = os.path.join(folder, '0.jpg').replace('\\', '/')
        file_1 = os.path.join(folder, file_1).replace('\\', '/')
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 1))
        exclude_list.add(file_0)
        exclude_list.add(file_1)

    for _ in tqdm(range(num_not_same)):
        dirs = [d for d in os.listdir(IMG_DIR_ALIGNED) if os.path.isdir(os.path.join(IMG_DIR_ALIGNED, d))]
        folders = random.sample(dirs, 2)
        while len([f for f in os.listdir(os.path.join(IMG_DIR_ALIGNED, folders[0])) if
                   f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1 or len(
            [f for f in os.listdir(os.path.join(IMG_DIR_ALIGNED, folders[1])) if
             f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
            folders = random.sample(dirs, 2)

        file_0 = folders[0] + '/' + '0.jpg'
        file_1 = pick_one_file(folders[1])
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 0))
        exclude_list.add(os.path.join(file_0))
        exclude_list.add(os.path.join(file_1))

    with open('data/test_pairs.txt', 'w') as file:
        file.writelines(out_lines)

    print(exclude_list)

    samples = get_data()
    filtered = []
    for item in samples:
        if item['img'] not in exclude_list:
            filtered.append(item)

    print(len(filtered))
    print(filtered[:10])

    with open(pickle_file, 'wb') as file:
        pickle.dump(filtered, file)
