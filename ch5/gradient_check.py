from typing import Any

import numpy as np
import numpy.typing as npt
from two_layer_net import TwoLayerNet

from dataset.mnist import load_mnist

NpArray = npt.NDArray[Any]


def main() -> None:
    (x_train, t_train), _ = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch: NpArray = x_train[:3]
    t_batch: NpArray = t_train[:3]

    grad_numerical: dict[str, NpArray] = network.numerical_gradient(x_batch, t_batch)
    grad_backprop: dict[str, NpArray] = network.gradient(x_batch, t_batch)

    for key, numerical in grad_numerical.items():
        diff = np.mean(np.abs(grad_backprop[key] - numerical))
        print(f"{key}: {diff}")


if __name__ == "__main__":
    main()
