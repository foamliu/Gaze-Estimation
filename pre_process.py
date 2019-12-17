import json
import os
import pickle

from tqdm import tqdm

from config import IMG_DIR, num_train

if __name__ == "__main__":
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]

    samples = []
    for filename in tqdm(files):
        full_path = os.path.join(IMG_DIR, filename)
        json_path = full_path.replace('.jpg', '.json')
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        look_vec = eval(data['eye_details']['look_vec'])
        look_vec = list(look_vec)[:3]
        iris_texture = data['eye_details']['iris_texture']
        pupil_size = float(data['eye_details']['pupil_size'])
        samples.append(
            {'filename': filename, 'look_vec': look_vec, 'pupil_size': pupil_size, 'iris_texture': iris_texture})
        # print(samples)

    print('num_samples: ' + str(len(samples)))

    train = samples[:num_train]
    val = samples[num_train:]

    with open('data/train.pkl', 'wb') as fp:
        pickle.dump(train, fp)

    with open('data/val.pkl', 'wb') as fp:
        pickle.dump(val, fp)

    print('num_train: ' + str(len(train)))
    print('num_val: ' + str(len(val)))
