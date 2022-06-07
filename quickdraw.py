# Modified from tensorflow.keras.datasets.mnist
"""QuickDraw handdrawn images dataset."""

import numpy as np

from keras.utils.data_utils import get_file


def load_data(path="quickdraw.npz"):
    """Loads the QuickDraw dataset.

    This is a dataset of 60,000 28x28 grayscale images of 10 animal species,
    along with a test set of 10,000 images.
    More info can be found at the
    [QuickDraw homepage](https://quickdraw.withgoogle.com/).

    Args:
      path: path where to cache the dataset locally
        (relative to `~/.keras/datasets`).

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of grayscale image data with shapes
      `(60000, 28, 28)`, containing the training data. Pixel values range
      from 0 to 255.

    **y_train**: uint8 NumPy array of class labels (integers in range 0-9)
      with shape `(60000,)` for the training data.

    **x_test**: uint8 NumPy array of grayscale image data with shapes
      (10000, 28, 28), containing the test data. Pixel values range
      from 0 to 255.

    **y_test**: uint8 NumPy array of class labels (integers in range 0-9)
      with shape `(10000,)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = quickdraw.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    ```
    """
    origin_folder = (
        "https://bioinf.nl/~davelangers/datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "quickdraw.npz",
        file_hash="155601d0dedea9fbcd95c6e07cfe42bbe45cc5f827b242483b6a6156b68bb5de",  # noqa: E501
    )
    with np.load(
        path, allow_pickle=True
    ) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)
