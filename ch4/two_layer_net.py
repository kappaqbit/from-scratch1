import numpy as np

import common.functions as fnc
from common.gradient import numerical_gradient


class TwolayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = fnc.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = fnc.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return fnc.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W(_W):
            return self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        a1 = np.dot(x, W1) + b1
        z1 = fnc.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = fnc.softmax(a2)

        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = fnc.sigmoid_grad(a1) * dz1
        grads["W1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads


if __name__ == "__main__":
    net = TwolayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["b1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b2"].shape)

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    print(net.accuracy(x, t))

    grads = net.numerical_gradient(x, t)

    print(grads["W1"].shape)
    print(grads["b1"].shape)
    print(grads["W2"].shape)
    print(grads["b2"].shape)
