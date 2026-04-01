import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import MLP_matrix as MPM
import MLP_jax as MPJ


def load_labels(path):
    """
    Load the labels from the dataset

    :param labels:
    """
    with open(path, 'rb') as f:
        # the first 8 bytes are metada
        labels = np.frombuffer(f.read(), dtype=jnp.uint8, offset=8)
    return jnp.array(labels)


def load_images(path):
    """
    Load the images form the dataset

    :param images:
    """
    with open(path, 'rb') as f:
        # the first 16 bytes are metada
        # The redimession give us a 28x28 matrices
        images = np.frombuffer(f.read(), dtype=jnp.uint8, offset=16).reshape(-1, 28, 28)
    return jnp.array(images)


def k_fold(X, y, k=5, seed=73):
    samples = len(X)
    index = jnp.arange(samples)
    key = jax.random.PRNGKey(seed)
    jax.random.shuffle(key, index)

    # divides the samples into folds
    folds = jnp.array_split(index, k)
    accuracies = []

    for i in range(k):
        # validation data
        id_val = folds[i]

        # training data
        id_train = jnp.concatenate([folds[j] for j in range(k) if j != i])

        X_train = X[id_train]
        y_train = y[id_train]
        X_val = X[id_val]
        y_val = y[id_val]

    # modelo.entrenar(X_train, y_train)
    # predicciones = modelo.predecir(X_val)

    # Calcular la precisión de este pliegue (acc_i)
    # acc_i = np.mean(predicciones == y_val)

    # temporal
    acc_i = 0.95
    accuracies.append(acc_i)

    # Calcular la media de las precisiones (acccv)
    acccv = jnp.mean(accuracies)

    return acccv, accuracies


def main():
    """
    Main function, call both MLP and charge the dataset
    """
    X_train = load_images('archive/train-images.idx3-ubyte')
    Y_train = load_labels('archive/train-labels.idx1-ubyte')
    X_test = load_images('archive/t10k-images.idx3-ubyte')
    Y_test = load_labels('archive/t10k-labels.idx1-ubyte')

    # print(f"X_train form: {X_train}")
    # print(f"y_train form: {Y_train}")


if __name__ == "__main__":
    main()
