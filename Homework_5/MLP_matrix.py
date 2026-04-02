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
        """
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
            activations_hidden = self.sigmoid(Z_hidden)
        
        # forward result on output layer
        Z_out = (activations_hidden @ self.params[f'W{total_layers}']) + self.params[f'b{total_layers}']
        self.neuron_state.append((activations_hidden, self.params[f'W{total_layers}'], Z_out)) # keep the state of each neuron
        self.y_out = self.softmax(Z_out)
        return self
    
    def backward_propagation(self):
        """
        """
        return self
    
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