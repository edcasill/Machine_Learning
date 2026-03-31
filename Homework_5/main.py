import os
import pandas as pd
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import MLP_matrix as MPM
import MLP_jax as MPJ


def load_labels(path):
    """
    Load the labels from the dataset

    :param labels: 
    """
    with open(path, 'rb') as f:
        # the first 8 bytes are metada
        labels = jnp.frombuffer(f.read(), dtype=jnp.uint8, offset=8)
    return labels


def load_images(path):
    """
    Load the images form the dataset

    :param images:
    """
    with open(path, 'rb') as f:
        # the first 16 bytes are metada
        # The redimession give us a 28x28 matrices
        images = jnp.frombuffer(f.read(), dtype=jnp.uint8, offset=16).reshape(-1, 28, 28)
    return images    


def main():
    """
    Main function, call both MLP and charge the dataset
    """
    X_train = load_images('archive/train-images.idx3-ubyte')
    Y_train = load_labels('archvive/train-labels.idx1-ubyte')
    X_test = load_images('archive/t10k-images.idx3-ubyte')
    Y_test = load_labels('archive/t10k-labels.idx1-ubyte')


if __name__ == "__main__":
    main()