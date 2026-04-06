import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """
    Use shannon impurity as splitting criterion
    """
    def __init__(self, max_depth=5, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.num_classes = 10

    def entropy(self, y):
        # Calculate P(w_i | t) 
        counts = jnp.bincount(y, length=self.num_classes)
        probs = counts / len(y)
        # Filter out zero probabilities to avoid log(0) issues
        probs = probs[probs > 0]
        # Sum( P * log2(P) )
        return -jnp.sum(probs * jnp.log2(probs))
    
    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape
        current_entropy = self.entropy(y)

        for feature in range(n_features):
            threshold = jnp.mean(X[:, feature]) # Use the mean of the feature as the threshold to split
            # Create a boolean mask for samples going to the left child
            left_mask = X[:, feature] <= threshold
            # The right child gets the exact opposite of the left mask
            # ~ reverses the boolean values
            right_mask = ~left_mask
            
            if not jnp.any(left_mask) or not jnp.any(right_mask):
                continue

            y_l, y_r = y[left_mask], y[right_mask] # Separate the labels according to the masks
            n_l, n_r = len(y_l), len(y_r) # Count the number of samples in each child node

            # Calculate the entropy for the left and right child nodes
            entropy_l, entropy_r = self.entropy(y_l), self.entropy(y_r)
            # Calculate Information Gain: Decrease in Impurity (Delta I)
            gain = current_entropy - (n_l / n_samples) * entropy_l - (n_r / n_samples) * entropy_r
            if gain > best_gain:
                best_gain = gain
                split_idx = feature
                split_thresh = threshold
        # Return the best feature and threshold found
        return split_idx, split_thresh

    def build_tree(self, X, y, depth=0):
        """
        bincount works by counting the occurrences of each class in the labels.
        The most common class is the one with the highest count.
        """
        n_samples = len(y)
        counts = jnp.bincount(y, length=self.num_classes) # Count the occurrences of each class
        most_common = jnp.argmax(counts)

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples or jnp.max(counts) == n_samples:
            return Node(value=most_common)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=most_common)
        
        # Create boolean masks to divide the dataset for the children
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)

    def predict(self, X):
        # Convert list of predictions to JAX array
        return jnp.array([self.predict_sample(x, self.root) for x in X])


def get_metrics(y_true, y_pred, num_classes=10):
    cm_1d = jnp.bincount(y_true * num_classes + y_pred, length=num_classes**2)
    cm = cm_1d.reshape((num_classes, num_classes))

    TP = jnp.diag(cm)
    FP = jnp.sum(cm, axis=0) - TP
    FN = jnp.sum(cm, axis=1) - TP
    epsilon = 1e-7
    
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    accuracy = jnp.trace(cm) / jnp.sum(cm)

    return accuracy, jnp.mean(precision), jnp.mean(recall), jnp.mean(f1)


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


def split_data(X, Y, train_size=0.85, seed=42):
    key = jax.random.PRNGKey(seed)
    num_samples = X.shape[0]
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    
    split_idx = int(num_samples * train_size)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def pca(X, componnents):
    """
    Reduce dimencionality to de dataset for the classification tree processing
    """
    # data center. Xc = X - mu(mean)
    mu = jnp.mean(X, axis=0)
    Xc = X - mu

    # covariance matrix
    cov_matrix = jnp.cov(Xc, rowvar=False)

    # eigen decomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)
    sorted_indices = jnp.argsort(eigenvalues)[::-1] # because linalg.eigh returns on ascendent order
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    V_k = sorted_eigenvectors[:, :componnents]
    X_pca = Xc @ V_k
    return X_pca, V_k, mu


def k_cross_validation_tree(X, Y, k=5):
    """
    Execute K-Cross Validation for the Decision Tree model
    """
    num_samples = X.shape[0]
    fold_size = num_samples // k
    all_metrics = []

    print(f"\nStarting {k}-Cross Validation for Decision Tree (PCA)...")
    key = jax.random.PRNGKey(42)
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, Y_val = X_shuffled[start:end], Y_shuffled[start:end]

        X_train = jnp.concatenate([X_shuffled[:start], X_shuffled[end:]], axis=0)
        Y_train = jnp.concatenate([Y_shuffled[:start], Y_shuffled[end:]], axis=0)

        tree = DecisionTree(max_depth=5)
        print(f"Training Fold {i+1}/{k}...", end=" ")

        tree.fit(X_train, Y_train)
        y_pred = tree.predict(X_val)

        acc, p, r, f1 = get_metrics(Y_val, y_pred)
        all_metrics.append({'acc': acc, 'p': p, 'r': r, 'f1': f1})
        print(f"| Accuracy: {acc:.4f}")

    return all_metrics


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

    k_dimension = 50
    X_train_pca, V_k, mu_train = pca(X_train, k_dimension)
    X_test_pca = (X_test - mu_train) @ V_k

    #
    cantidad_datos = 10000 # for a faster execution
    key_sample = jax.random.PRNGKey(99)
    indices_sample = jax.random.permutation(key_sample, jnp.arange(X_train_pca.shape[0]))[:cantidad_datos]
    
    X_train_rapido = X_train_pca[indices_sample]
    Y_train_rapido = Y_train[indices_sample]
    #

    results = k_cross_validation_tree(X_train_rapido, Y_train_rapido, k=5)
    print("\n>>> K-CROSS REPORT (DECISION TREE + PCA) <<<")
    print(f"Mean Accuracy  : {sum(r['acc'] for r in results) / len(results):.4f}")
    print(f"Mean Precision : {sum(r['p'] for r in results) / len(results):.4f}")
    print(f"Mean Recall    : {sum(r['r'] for r in results) / len(results):.4f}")
    print(f"Mean F1-Score  : {sum(r['f1'] for r in results) / len(results):.4f}")

    print("\nTraining Final Decision Tree for Test Set...")
    final_tree = DecisionTree(max_depth=5)
    final_tree.fit(X_train_pca, Y_train)
    y_pred_final = final_tree.predict(X_test_pca)
    
    acc_test, p_test, r_test, f1_test = get_metrics(Y_test, y_pred_final)

    print("\nFINAL K-CROSS REPORT (DECISION TREE + PCA)")
    print(f"Mean Accuracy  : {acc_test:.4f}")
    print(f"Mean Precision : {p_test:.4f}")
    print(f"Mean Recall    : {r_test:.4f}")
    print(f"Mean F1-Score  : {f1_test:.4f}")


if __name__ == "__main__":
    main()
