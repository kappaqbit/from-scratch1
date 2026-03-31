import matplotlib.pylab as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    X = np.arange(-5.0, 5.0, 0.1)
    Y = relu(X)

    plt.plot(X, Y)
    plt.ylim(-1.0, 5.5)
    plt.show()
