import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from MLP_matrix import Multilayer_Perceptron_Matrix as MPM
from MLP_jax import Multilayer_Perceptron_JAX as MPJ
import k_cross as kx


def load_labels(path):
    """
    Load the labels from the dataset

    Args:
        path (_type_): path to the labels

    Returns:
        labels: jnp array for matrix operation
    """
    with open(path, 'rb') as f:
        # the first 8 bytes are metada
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return jnp.array(labels)


def load_images(path):
    """
    Load the images form the dataset

    Args:
        path (_type_): path to the images file

    Returns:
        images: matrix with the images on a size 28x28
    """
    with open(path, 'rb') as f:
        # the first 16 bytes are metada
        # The redimession give us a 28x28 matrices
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    return jnp.array(images)


def preprocess(images):
    """
    flat the images and normalice the values

    Args:
        images (_type_): training or validation images to train/validate the model

    Returns:
        images_normalized: the normaliced values of the images
    """
    images_flattened = images.reshape(images.shape[0], -1)
    images_normalized = images_flattened.astype(jnp.float32) / 255.0
    return images_normalized


def main():
    """
    Main function, call both MLP and charge the dataset
    """
    X_train = load_images('archive/train-images.idx3-ubyte')
    Y_train = load_labels('archive/train-labels.idx1-ubyte')
    X_test = load_images('archive/t10k-images.idx3-ubyte')
    Y_test = load_labels('archive/t10k-labels.idx1-ubyte')

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # print(f"X_train form: {X_train}")
    # print(f"y_train form: {Y_train}")
    matrix_mlp = MPM(X_train)
    matrix_params = matrix_mlp.fit_mlp_matrix(X_train, Y_train, 1000, 0.1)


if __name__ == "__main__":
    main()
