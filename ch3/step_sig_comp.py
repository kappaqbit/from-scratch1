import matplotlib.pylab as plt
import numpy as np
from sigmoid_function import sigmoid
from step_function import step_function

if __name__ == "__main__":
    X = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(X)
    y2 = step_function(X)

    plt.plot(X, y1)
    plt.plot(X, y2, "k--")
    plt.ylim(-0.1, 1.1)
    plt.show()
