import numpy as np
import numpy.typing as npt
from two_layer_net import TwoLayerNet

from dataset.mnist import load_mnist

FloatArray = npt.NDArray[np.float64]


def main() -> None:
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=True
    )

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num: int = 20000
    train_size: int = int(x_train.shape[0])
    batch_size: int = 100
    learning_rate: float = 0.1

    train_loss_list: list[float] = []
    train_acc_list: list[float] = []
    test_acc_list: list[float] = []

    iter_per_epoch: int = max(train_size // batch_size, 1)

    for i in range(iters_num):
        batch_mask: npt.NDArray[np.int_] = np.random.choice(train_size, batch_size)
        x_batch: FloatArray = x_train[batch_mask]
        t_batch: FloatArray = t_train[batch_mask]

        grads: dict[str, FloatArray] = network.gradient(x_batch, t_batch)

        for key in network.params:
            network.params[key] -= learning_rate * grads[key]

        loss: float = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc: float = network.accuracy(x_train, t_train)
            test_acc: float = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            print(
                f"iter: {i:>5} | train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}"
            )


if __name__ == "__main__":
    main()
