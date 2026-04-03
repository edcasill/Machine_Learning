import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


class Multilayer_Perceptron_Matrix():
    """
    Multilayer pereptron ussing the classic matrix implementation
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

    def forward_propagation(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.neuron_state = []
        activations_hidden = X
        # this is because I'm keeping 2 values for each layer (bias an weigth),
        # obtaining the int number from the divison give me the quantity of real layers, not it's elements
        total_layers = len(self.params) // 2

        # forward result on hidden layer
        for i in range(1, total_layers): 
            Z_hidden = (activations_hidden @ self.params[f'W{i}']) + self.params[f'b{i}']
            self.neuron_state.append((activations_hidden, self.params[f'W{i}'], Z_hidden)) # keep the state of each neuron
            # activations_hidden = self.sigmoid(Z_hidden)
            activations_hidden = self.relu(Z_hidden)

        # forward result on output layer
        Z_out = (activations_hidden @ self.params[f'W{total_layers}']) + self.params[f'b{total_layers}']
        self.neuron_state.append((activations_hidden, self.params[f'W{total_layers}'], Z_out)) # keep the state of each neuron
        self.y_out = self.softmax(Z_out)
        return self
    
    def backward_propagation(self, Y):
        """
        Backward propagation error from output layer to hidden layer

        Args:
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.gradients = {} # keep the gradients from W and Bias
        total_transformed_layers = len(self.neuron_state)
        batch_size = Y.shape[0] # amount of images evualuated on the forward propagation
        self.Y = self.onehot_encoding(Y, self.layers[-1])

        # output layer error
        prev_activation, W, Z = self.neuron_state[-1] # -1 means the last item
        dZ = self.y_out - self.Y

        self.gradients[f'dW{total_transformed_layers}'] = (prev_activation.T @ dZ) / batch_size
        self.gradients[f'db{total_transformed_layers}'] = jnp.sum(dZ, axis=0, keepdims=True) / batch_size

        # error propagation to hidden layer
        for i in reversed(range(total_transformed_layers - 1)):
            prev_activation, W, Z = self.neuron_state[i]

            W_next = self.neuron_state[i+1][1]
            dZ_next = dZ

            # propagated error * activation derivative
            dActivation = dZ_next @ W_next.T
            # dZ = dActivation * self.sigmoid_derivative(Z)
            dZ = dActivation * self.relu_derivada(Z)

            # gradient on actual layer
            self.gradients[f'dW{i+1}'] = (prev_activation.T @ dZ) / batch_size
            self.gradients[f'db{i+1}'] = jnp.sum(dZ, axis=0, keepdims=True) / batch_size
        return self
    
    def update_parameters(self, learning_rate):
        total_layers = len(self.params) // 2

        for i in range(1, total_layers + 1):
            self.params[f'W{i}'] -= learning_rate * self.gradients[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * self.gradients[f'db{i}']
        return self
    
    def fit_mlp_matrix(self, X, Y, epochs, learning_rate, seed=73):
        for epoch in range(epochs):
            # forward prop
            self.forward_propagation(X)

            # backward prop
            self.backward_propagation(Y)

            # update data
            self.update_parameters(learning_rate)

            if epoch % 10 == 0:
                predictions = jnp.argmax(self.y_out, axis=1)
                precission = jnp.mean(predictions == Y)
                print(f"Epoch {epoch} | Precisión: {precission:.4f}")
        return self.params
    
    @staticmethod
    def onehot_encoding(Y, out_classes):
        """
        Apply the one-hot encoding ussing jnp.eye, wich generate an identity matrix with the size of the number of
        output classes, indexing the labels vector in it

        Args:
            Y (_type_): labels vector
            out_classes (_type_): labels one hot

        Returns:
            _type_: _description_
        """
        return jnp.eye(out_classes)[Y]
    
    # activation functions
    @staticmethod
    def sigmoid(Z):
        """
        Sigmoid activation funtion

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + jnp.exp(-Z))

    @staticmethod
    def sigmoid_derivative(Z):
        """
        Sigmoid derivative

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (jnp.exp(-Z)) / (1 + jnp.exp(-Z))**2

    @staticmethod
    def softmax(Z):
        """
        Lineal transformation, previous layer multiplied by weigths plus the bias

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        exp_Z = jnp.exp(Z - jnp.max(Z, axis=1, keepdims=True))
        return exp_Z / jnp.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def relu(Z):
        """
        Rectified Linear Unit for non linearity into the net

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return jnp.maximum(0, Z)

    @staticmethod
    def relu_derivada(Z):
        """
        Relu derivate, is 1 if Z > 0, 0 otherwise

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (Z > 0).astype(jnp.float32)