import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_pmf_distribution(name):
    print('computing {}...'.format(name))

    c = dict()
    for sample in tqdm(samples):
        type = sample[name]
        if type in c:
            c[type] += 1
        else:
            c[type] = 1

    x = c.keys()
    y = list(c.values())
    y = np.array(y)
    y = y / y.sum()
    y = list(y)
    plt.bar(x, y, color='blue')
    plt.title(name)

    plt.savefig('images/{}_dist.png'.format(name))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    with open('data/train.pkl', 'rb') as fp:
        samples = pickle.load(fp)

    compute_pmf_distribution('iris_texture')
