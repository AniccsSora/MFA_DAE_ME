import matplotlib.pyplot as plt
import numpy as np


a = np.array([100, 8, 14, 22, 72, 17, 8, 5, 5, 5, 8, 8, 4, 1, 4, 2, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0])
b = np.array([150, 1, 108, 19, 22, 3, 8, 5, 7, 8, 6, 5, 10, 14, 14, 14, 13, 12, 12, 12, 12, 9, 5, 2, 8, 19, 16])
c = np.array([150, 18, 30, 87, 11, 30, 7, 17, 19, 8, 2, 7, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def get_bar(bar):
    bar = bar.reshape(-1, 1)
    return bar[::-1]


if __name__ == "__main__":

    bar = get_bar(c)  ##

    print(bar.shape)
    plt.figure(dpi=300)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.grid(False)
    plt.tight_layout()
    plt.imshow(bar, cmap='plasma')
    plt.savefig("fffufufufuf.png")
    plt.show()


