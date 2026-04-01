import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


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