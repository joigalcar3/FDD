#!/usr/bin/env python3
"""
Fault Detection and Diagnosis model that uses a Deep NN for the sensor fusion, capturing temporal information
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"


# Imports
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Input


class DeepNeuralFusionModel:
    """
    Model that fuses the camera and sensor data using a simple Deep NN
    """
    def __init__(self, n_layers, filters, num_classes, layers_activations="tanh", kernel_initializer="glorot_uniform"):
        """
        Initializes the class by creating each of the layers.
        :param n_layers: number of layers that combine an LSTM and a Batch Normalization layer
        :param filters: number of filters
        :param num_classes: number of output classes
        :param layers_activations: the activation functions used in the NN layers
        :param kernel_initializer: the initialization methods for the NN weights
        """
        self.layers = []
        for i in range(n_layers):
            # When BN is used: Usually a neural network has a weight matrix and bias vector as parameters.
            # However, we no longer need the bias because the batch normalization already shifts the data!
            # https://www.nathanieldake.com/Deep_Learning/02-Modern_Deep_Learning-11-BatchNormalization.html
            # This is only applicable when BN is done before the activation function, which here is not the case
            Dense_layer = Dense(filters, activation=layers_activations, kernel_initializer=kernel_initializer,
                                name=f"dense_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.layers.extend([Dense_layer, BN])
        self.dense = Dense(num_classes, activation="softmax", name="classifier")

    def model(self, input_shape):
        """
        Concatenates the layers to create the model architecture
        :param input_shape: the array shape of the model
        :return: the model object
        """
        input_layer = Input(shape=(*input_shape,))
        x = input_layer
        for layer in self.layers:
            x = layer(x)
        x = self.dense(x)
        output = x
        return Model(inputs=input_layer, outputs=output)
