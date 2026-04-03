import jax
import jax.numpy as jnp


class Multilayer_Perceptron_JAX():
    """
    Multilayer pereptron ussing the JAX forward implementation with autodiff
    """
    def __init__(self, X, layers=[0, 128, 10], seed=73):
        key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(key, len(layers) - 1)
        layers[0] = X.shape[1]  # input layer
        self.layers = layers
        self.params = {}

        # this is implemented if in a future we want to add more hiden layers
        for i in range(len(self.layers) - 1):
            dim_in = self.layers[i]
            dim_out = self.layers[i+1]
            
            # Weights and bias initialization for each hidden layer and the output layer
            self.params[f'W{i+1}'] = jax.random.normal(self.keys[i], (dim_in, dim_out)) * jnp.sqrt(2.0 / dim_in)
            self.params[f'b{i+1}'] = jnp.zeros((1, dim_out))

    @staticmethod
    def forward_propagation(params, X):
        """_summary_

        Args:
            params (_type_): _description_
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        activations_hidden = X
        total_layers = len(params) // 2

        for i in range(1, total_layers): 
            Z_hidden = (activations_hidden @ params[f'W{i}']) + params[f'b{i}']
            activations_hidden = jnp.maximum(0, Z_hidden)

        # forward result on output layer
        Z_out = (activations_hidden @ params[f'W{total_layers}']) + params[f'b{total_layers}']
        exp_z = jnp.exp(Z_out - jnp.max(Z_out, axis=1, keepdims=True))
        y_out = exp_z / jnp.sum(exp_z, axis=1, keepdims=True)
        return y_out
    
    @staticmethod
    def loss(params, X, Y_hot):
        """
        Cost function (Cross-Entropy)
        """
        predictions = Multilayer_Perceptron_JAX.forward_propagation(params, X)
        error = -jnp.sum(Y_hot * jnp.log(predictions + 1e-8)) / X.shape[0]
        return error
    
    def fit_mlp_jax(self, X, Y, epochs, learning_rate):
        """
        
        """
        # One-hot encoding
        Y_hot = jnp.eye(self.layers[-1])[Y]
        
        # gradient function
        grad_fn = jax.grad(self.loss)

        for epoch in range(epochs):
            # get all grafients evaluating grad_fn
            gradients = grad_fn(self.params, X, Y_hot)

            # update parameters
            total_layers = len(self.params) // 2
            for i in range(1, total_layers + 1):
                self.params[f'W{i}'] -= learning_rate * gradients[f'W{i}']
                self.params[f'b{i}'] -= learning_rate * gradients[f'b{i}']

            if epoch % 10 == 0:
                y_out = self.forward_propagation(self.params, X)
                predictions = jnp.argmax(y_out, axis=1)
                precision = jnp.mean(predictions == Y)
                print(f"Epoch {epoch} | Precissionn: {precision:.4f}")

        return self.params