import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from MLP_matrix import Multilayer_Perceptron_Matrix as MPM
from MLP_jax import Multilayer_Perceptron_JAX as MPJ
import matplotlib.pyplot as plt


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


def plot_cm(ax, cm, fig, title):
    """
    Plot the confusion matrix

    Args:
        ax (_type_): _description_
        cm (_type_): _description_
        fig (_type_): _description_
        title (_type_): _description_
    """
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    tick_marks = jnp.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Real Label')

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")


def print_table(p, r, f1, nombre):
    """
    Print the table of precision, recall anf F1-score

    Args:
        p (_type_): _description_
        r (_type_): _description_
        f1 (_type_): _description_
        nombre (_type_): _description_
    """
    print(f"\n{nombre}")
    print(f"{'Class':<6} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 45)
    for i in range(10):
        print(f"  {i:<4} | {p[i]:.4f}     | {r[i]:.4f}     | {f1[i]:.4f}")

    print("-" * 45)
    print(f"Macro  | {jnp.mean(p):.4f}     | {jnp.mean(r):.4f}     | {jnp.mean(f1):.4f}")


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
    matrix_params, matrix_loss_history = matrix_mlp.fit_mlp_matrix(X_train, Y_train, 1000, 0.1)
    cm_mat, p_mat, r_mat, f1_mat = matrix_mlp.get_metrics(X_test, Y_test)

    print('_'*60)

    jax_mlp = MPJ(X_train)
    jax_result, jax_loss_history = jax_mlp.fit_mlp_jax(X_train, Y_train, 1000, 0.1)
    cm_jax, p_jax, r_jax, f1_jax = jax_mlp.get_metrics(X_test, Y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))


    plot_cm(ax1, cm_mat, fig, "Matrix Backpropagation")
    plot_cm(ax2, cm_jax, fig, "Autodiff Backpropagation (JAX)")
    plt.tight_layout()
    plt.savefig("Confusion matrices.png")

    print_table(p_mat, r_mat, f1_mat, "Matrix")
    print_table(p_jax, r_jax, f1_jax, "Autodiff")

    plt.figure(figsize=(10, 5))
    plt.plot(matrix_loss_history, label='Matrix Backprop', linewidth=2)
    plt.plot(jax_loss_history, label='JAX Autodiff', linestyle='dashed', linewidth=2)
    plt.title('Convergence Rate: Matrix vs Autodiff')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("Convergence rate.png")


if __name__ == "__main__":
    main()
