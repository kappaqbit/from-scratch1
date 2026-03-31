import numpy as np
import PIL.ImageShow
from PIL import Image

from dataset.mnist import load_mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == "__main__":
    PIL.ImageShow.register(PIL.ImageShow.EogViewer(), 0)

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=False, flatten=True, one_hot_label=False
    )

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)
