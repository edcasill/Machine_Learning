import os
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


class Multilayer_Perceptron_Matrix():
    """
    Multilayer pereptron ussing the classic matrix implementation
    """
    def __init__(self, X, h_layers=1, seed=73):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, len(h_layers) - 1)
        self.params = {}

        for i in range(len(h_layers) - 1):
            dim_in = h_layers[i]
            dim_out = h_layers[i+1]
            
            # Inicialización He
            self.params[f'W{i+1}'] = jax.random.normal(keys[i], (dim_in, dim_out)) * jnp.sqrt(2.0 / dim_in)
            self.params[f'b{i+1}'] = jnp.zeros((1, dim_out))

    def forward_propagation():
        """
        """
    
    def backward_propafation():
        """
        """
    
    # activation functions

    def sigmoid(Z):
        """
        Sigmoid activation funtion

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 1 / (1 + jnp.exp(-Z))
    
    def sigmoid_derivative(Z):
        """
        Sigmoid derivative

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (jnp.exp(-Z)) / (1 + jnp.exp(-Z))**2
    
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
    
    def relu(Z):
        """
        Rectified Linear Unit for non linearity into the net

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return jnp.maximum(0, Z)

    def relu_derivada(Z):
        """
        Relu derivate, is 1 if Z > 0, 0 otherwise

        Args:
            Z (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (Z > 0).astype(jnp.float32)