from typing import Any

import numpy as np
import numpy.typing as npt

from common.gradient import numerical_gradient
from common.layers import Affine, Relu, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_std: float = 0.01,
    ) -> None:
        self.params: dict[str, npt.NDArray[Any]] = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers: dict[str, Any] = {}
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x: npt.NDArray[Any], t: npt.NDArray[Any]) -> float:
        y = self.predict(x)
        return float(self.lastLayer.forward(y, t))

    def accuracy(self, x: npt.NDArray[Any], t: npt.NDArray[Any]) -> float:
        y = self.predict(x)
        y_argmax = np.argmax(y, axis=1)
        t_artmax = np.argmax(t, axis=1) if t.ndim != 1 else t

        accuracy_val: float = np.sum(y_argmax == t_artmax) / float(x.shape[0])
        return accuracy_val

    def numerical_gradient(
        self, x: npt.NDArray[Any], t: npt.NDArray[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        def loss_W(_W: npt.NDArray[Any]) -> float:
            return self.loss(x, t)

        grads: dict[str, npt.NDArray[Any]] = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(
        self, x: npt.NDArray[Any], t: npt.NDArray[Any]
    ) -> dict[str, npt.NDArray[Any]]:
        self.loss(x, t)
        dout: float | npt.NDArray[Any] = 1.0
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads: dict[str, npt.NDArray[Any]] = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads
