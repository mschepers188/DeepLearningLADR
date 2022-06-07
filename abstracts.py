# Modified from tensorflow.keras.datasets.reuters
"""PubMed abstracts journal classification dataset."""

import json

import numpy as np

from tensorflow import keras

from keras.preprocessing.sequence import _remove_long_seq
from keras.utils.data_utils import get_file

# isort: off
from tensorflow.python.platform import tf_logging as logging


def load_data(
    path="abstracts.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs,
):
    """Loads the PubMed abstracts classification dataset.

    This is a dataset of 4,908 abstracts from PubMed, labeled over 2 journals.

    Each abstract is encoded as a list of word indexes (integers).
    For convenience, words are indexed by overall frequency in the dataset,
    so that for instance the integer "3" encodes the 3rd most frequent word in
    the data. This allows for quick filtering operations such as:
    "only consider the top 10,000 most
    common words, but eliminate the top 20 most common words".

    As a convention, "0" does not stand for a specific word, but instead is used
    to encode any unknown word.

    Args:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: integer or None. Words are
          ranked by how often they occur (in the training set) and only
          the `num_words` most frequent words are kept. Any less frequent word
          will appear as `oov_char` value in the sequence data. If None,
          all words are kept. Defaults to None, so all words are kept.
      skip_top: skip the top N most frequently occurring words
          (which may not be informative). These words will appear as
          `oov_char` value in the dataset. Defaults to 0, so no words are
          skipped.
      maxlen: int or None. Maximum sequence length.
          Any longer sequence will be truncated. Defaults to None, which
          means no truncation.
      test_split: Float between 0 and 1. Fraction of the dataset to be used
        as test data. Defaults to 0.2, meaning 20% of the dataset is used as
        test data.
      seed: int. Seed for reproducible data shuffling.
      start_char: int. The start of a sequence will be marked with this
          character. Defaults to 1 because 0 is usually the padding character.
      oov_char: int. The out-of-vocabulary character.
          Words that were cut out because of the `num_words` or
          `skip_top` limits will be replaced with this character.
      index_from: int. Index actual words with this index and higher.
      **kwargs: Used for backwards compatibility.

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: lists of sequences, which are lists of indexes
      (integers). If the num_words argument was specific, the maximum
      possible index value is `num_words - 1`. If the `maxlen` argument was
      specified, the largest possible sequence length is `maxlen`.

    **y_train, y_test**: lists of integer labels (1 or 0).

    Note: The 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    # Legacy support
    if "nb_words" in kwargs:
        logging.warning(
            "The `nb_words` argument in `load_data` "
            "has been renamed `num_words`."
        )
        num_words = kwargs.pop("nb_words")
    if kwargs:
        raise TypeError(f"Unrecognized keyword arguments: {str(kwargs)}")

    origin_folder = (
        "https://bioinf.nl/~davelangers/datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "abstracts.npz",
        file_hash="aa335f184316e29c9d42025a06e7483440af8ae88e492e4b37c6c8b6e11e73e1",  # noqa: E501
    )
    with np.load(
        path, allow_pickle=True
    ) as f:  # pylint: disable=unexpected-keyword-arg
        xs, labels = f["x"], f["y"]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(xs))
    rng.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]
    
    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)

    if not num_words:
        num_words = max(max(x) for x in xs)

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [
            [w if skip_top <= w < num_words else oov_char for w in x]
            for x in xs
        ]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = int(len(xs) * (1 - test_split))
    x_train, y_train = np.array(xs[:idx], dtype="object"), np.array(
        labels[:idx]
    )
    x_test, y_test = np.array(xs[idx:], dtype="object"), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path="abstracts_word_index.json"):
    """Retrieves a dict mapping words to their index in the abstracts dataset.

    Args:
        path: where to cache the data (relative to `~/.keras/dataset`).

    Returns:
        The word index dictionary. Keys are word strings, values are their
        index.
    """
    origin_folder = (
        "https://bioinf.nl/~davelangers/datasets/"
    )
    path = get_file(
        path,
        origin=origin_folder + "abstracts_word_index.json",
        file_hash="6cd750800af4c0f5ec608c780f9e22174f9bc812ed5e7ee933c3535324faa2b6",
    )
    with open(path) as f:
        return json.load(f)
