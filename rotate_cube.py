import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

if __name__ == '__main__':

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    for i in range(10):
        item = data[i]
        x, y, z = item['out'][0], item['out'][1], item['out'][2]

        soa = np.array([[0, 0, 0, x, y, z]])

        X, Y, Z, U, V, W = zip(*soa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # plt.show()

        plt.savefig("images/{}_angle.jpg".format(i))

