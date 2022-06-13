# Modified from tensorflow.keras.datasets.mnist
"""UTKface pictures dataset."""

import numpy as np

from keras.utils.data_utils import get_file


def load_data(path="faces.npz"):
    """Loads the UTKface dataset.

    This is a dataset of 10,000 64x64 color pictures of human faces with
    corresponding labels, along with a test set of 2,500 images. Labels
    refer to gender (0=male; 1=female), age (0=young; 1=old), and
    ethnicity (0=white; 1=black), respectively.
    More info can be found at the
    [UTKface homepage](https://susanqq.github.io/UTKFace/).

    Args:
      path: path where to cache the dataset locally
        (relative to `~/.keras/datasets`).

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of color picture data with shapes
      `(10000, 64, 64)`, containing the training data. Pixel values range
      from 0 to 255.

    **y_train**: uint8 NumPy array of integer labels with shape
      `(10000, 3)` for the training data.

    **x_test**: uint8 NumPy array of color picture data with shapes
      (2500, 64, 64), containing the test data. Pixel values range
      from 0 to 255.

    **y_test**: uint8 NumPy array of integer labels with shape
      `(2000, 3)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = faces.load_data()
    assert x_train.shape == (10000, 64, 64)
    assert x_test.shape == (2500, 64, 64)
    assert y_train.shape == (10000, 3)
    assert y_test.shape == (2500, 3)
    ```
    """
    origin_folder = (
        "https://bioinf.nl/~davelangers/datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "faces.npz",
        file_hash="2553b7000bbae7329e8d8e701921f825082f14973b597efbc7f4025f1c2a3982",  # noqa: E501
    )
    with np.load(
        path, allow_pickle=True
    ) as f:  # pylint: disable=unexpected-keyword-arg
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)
