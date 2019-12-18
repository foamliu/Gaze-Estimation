import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

if __name__ == '__main__':

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    for i in range(10):
        item = data[i]
        x, y, z = item['out']['look_vec'][0], item['out']['look_vec'][1], item['out']['look_vec'][2]

        soa = np.array([[0, 0, 0, x, -y, z]])

        X, Y, Z, U, V, W = zip(*soa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        ax.view_init(elev=-90, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # plt.show()

        plt.savefig("images/{}_angle.jpg".format(i))
        # break

