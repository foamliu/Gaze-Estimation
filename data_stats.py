import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm import tqdm


def compute_type_distribution(name):
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


def compute_value_distribution(name):
    print('computing {}...'.format(name))

    x = []
    for sample in tqdm(samples):
        value = sample[name]
        x.append(value)

    bins = np.linspace(-1, 1, 100)

    # the histogram of the data
    plt.hist(x, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu = np.mean(x)
    sigma = np.std(x)
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('{}'.format(name))
    plt.ylabel('{} distribution'.format(name))
    plt.title('Histogram: mu={:.4f}, sigma={:.4f}'.format(mu, sigma))

    plt.savefig('images/{}_dist.png'.format(name))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    with open('data/train.pkl', 'rb') as fp:
        samples = pickle.load(fp)

    compute_type_distribution('iris_texture')
    compute_value_distribution('pupil_size')
