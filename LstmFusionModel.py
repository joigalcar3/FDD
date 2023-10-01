#!/usr/bin/env python3
"""
Fault Detection and Diagnosis model that uses LSTMs for the sensor fusion, capturing temporal information
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
from keras.layers import LSTM, Dense, BatchNormalization, Input


class LstmFusionModel:
    """
    Model that fuses the camera and the sensor information with an LSTM. The model is created with the Keras functional
    API, hence the name.
    """
    def __init__(self, n_layers, filters, num_classes):
        """
        Initializes the class by creating each of the layers.
        :param n_layers: number of layers that combine an LSTM and a Batch Normalization layer
        :param filters: number of filters
        :param num_classes: number of output classes
        """
        self.layers = []
        for i in range(n_layers):
            LSTM_Batch_layer = LSTM(filters, return_sequences=True, name=f"LSTM_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.layers.extend([LSTM_Batch_layer, BN])
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
