import jax
import jax.numpy as jnp
import numpy as np

def load_labels(path):
    with open(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return jnp.array(labels)

def load_images(path):
    with open(path, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    return jnp.array(images)

def preprocess(images):
    images_flattened = images.reshape(images.shape[0], -1)
    images_normalized = images_flattened.astype(jnp.float32) / 255.0
    return images_normalized

class Multilayer_Perceptron_JAX():
    def __init__(self, X, layers=[0, 128, 10], seed=73):
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, len(layers) - 1)
        layers[0] = X.shape[1] 
        self.layers = layers
        self.params = {}

        for i in range(len(self.layers) - 1):
            dim_in = self.layers[i]
            dim_out = self.layers[i+1]
            self.params[f'W{i+1}'] = jax.random.normal(self.keys[i], (dim_in, dim_out)) * jnp.sqrt(2.0 / dim_in)
            self.params[f'b{i+1}'] = jnp.zeros((1, dim_out))

    @staticmethod
    def forward_propagation(params, X):
        activations_hidden = X
        total_layers = len(params) // 2

        for i in range(1, total_layers): 
            Z_hidden = (activations_hidden @ params[f'W{i}']) + params[f'b{i}']
            activations_hidden = jnp.maximum(0, Z_hidden)

        Z_out = (activations_hidden @ params[f'W{total_layers}']) + params[f'b{total_layers}']
        exp_z = jnp.exp(Z_out - jnp.max(Z_out, axis=1, keepdims=True))
        y_out = exp_z / jnp.sum(exp_z, axis=1, keepdims=True)
        return y_out
    
    @staticmethod
    def loss(params, X, Y_hot):
        predictions = Multilayer_Perceptron_JAX.forward_propagation(params, X)
        error = -jnp.sum(Y_hot * jnp.log(predictions + 1e-8)) / X.shape[0]
        return error
    
    def get_metrics(self, X_test, Y_test):
        pred_probs = self.forward_propagation(self.params, X_test)
        y_pred = jnp.argmax(pred_probs, axis=1)
        num_classes = self.layers[-1]

        cm_1d = jnp.bincount(Y_test * num_classes + y_pred, length=num_classes**2)
        cm = cm_1d.reshape((num_classes, num_classes))
        
        TP = jnp.diag(cm)
        FP = jnp.sum(cm, axis=0) - TP
        FN = jnp.sum(cm, axis=1) - TP
        epsilon = 1e-7
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        return cm, precision, recall, f1
    
    def fit_mlp_jax(self, X, Y, epochs, learning_rate, verbose=True):
        Y_hot = jnp.eye(self.layers[-1])[Y]
        grad_fn = jax.grad(self.loss)
        loss_history = []

        for epoch in range(epochs):
            gradients = grad_fn(self.params, X, Y_hot)
            total_layers = len(self.params) // 2
            
            for i in range(1, total_layers + 1):
                self.params[f'W{i}'] -= learning_rate * gradients[f'W{i}']
                self.params[f'b{i}'] -= learning_rate * gradients[f'b{i}']

            if verbose and epoch % 10 == 0:
                current_loss = self.loss(self.params, X, Y_hot)
                loss_history.append(current_loss)
                y_out = self.forward_propagation(self.params, X)
                predictions = jnp.argmax(y_out, axis=1)
                prec = jnp.mean(predictions == Y)
                print(f"  Epoch {epoch:3d} | Precision: {prec:.4f} | Loss: {current_loss:.4f}")

        return self.params, loss_history

# -------------------------------------------------------------------------
# K-FOLD VALIDATION LOGIC
# -------------------------------------------------------------------------

def k_cross_validation(X, Y, k=5, epochs=50, lr=0.1):
    num_samples = X.shape[0]
    fold_size = num_samples // k
    all_metrics = []

    print(f"\nStarting K-Cross Validation with {k} folds")

    key = jax.random.PRNGKey(15)
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        start, end = i * fold_size, (i + 1) * fold_size
        
        X_val_fold = X_shuffled[start:end]
        Y_val_fold = Y_shuffled[start:end]
        
        X_train_fold = jnp.concatenate([X_shuffled[:start], X_shuffled[end:]], axis=0)
        Y_train_fold = jnp.concatenate([Y_shuffled[:start], Y_shuffled[end:]], axis=0)
        
        # new instance of the model for each fold
        model_fold = Multilayer_Perceptron_JAX(X_train_fold, seed=i)
        
        # we use verbose=False to avoid an ocerflow on the console on each fold
        model_fold.fit_mlp_jax(X_train_fold, Y_train_fold, epochs, lr, verbose=False)
        
        cm, p, r, f1 = model_fold.get_metrics(X_val_fold, Y_val_fold)
        accuracy = jnp.trace(cm) / jnp.sum(cm)
        
        all_metrics.append({
            'accuracy': accuracy,
            'precision': jnp.mean(p),
            'recall': jnp.mean(r),
            'f1': jnp.mean(f1)
        })
        print(f"Accuracy de validación: {accuracy:.4f}")

    return all_metrics

def main():
    X_train = preprocess(load_images('archive/train-images.idx3-ubyte'))
    Y_train = load_labels('archive/train-labels.idx1-ubyte')
    X_test = preprocess(load_images('archive/t10k-images.idx3-ubyte'))
    Y_test = load_labels('archive/t10k-labels.idx1-ubyte')

    # just use k-fold
    results = k_cross_validation(X_train, Y_train, k=5, epochs=100, lr=0.1)
    
    print("\n>>> K-FOLD Results(Mean) <<<")
    print(f"Mean Accuracy  : {sum(r['accuracy'] for r in results) / len(results):.4f}")
    print(f"Mean Precision : {sum(r['precision'] for r in results) / len(results):.4f}")
    print(f"Mean Recall    : {sum(r['recall'] for r in results) / len(results):.4f}")
    print(f"Mean F1-Score  : {sum(r['f1'] for r in results) / len(results):.4f}")

    print("\n==================================================")
    print("FINAL EVALUATION: TEST DATASET")
    print("==================================================")
    
    final_model = Multilayer_Perceptron_JAX(X_train, seed=99)
    # we print the progress of the model
    final_model.fit_mlp_jax(X_train, Y_train, epochs=100, learning_rate=0.1, verbose=True)

    cm_test, p_test, r_test, f1_test = final_model.get_metrics(X_test, Y_test)
    accuracy_test = jnp.trace(cm_test) / jnp.sum(cm_test)

    print("\n>>> MÉTRICAS FINALES (TEST SET) <<<")
    print(f"Test Accuracy  : {accuracy_test:.4f}")
    print(f"Test Precision : {jnp.mean(p_test):.4f}")
    print(f"Test Recall    : {jnp.mean(r_test):.4f}")
    print(f"Test F1-Score  : {jnp.mean(f1_test):.4f}")

if __name__ == "__main__":
    main()
