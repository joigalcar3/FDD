import tensorflow as tf
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Input, Layer, Reshape, Lambda, GlobalAveragePooling3D
from keras.models import Sequential
import pydot
import numpy as np

# TODO: GlobalMaxPooling2D


class ImuModelConvLstm(tf.keras.Model):
    def __init__(self, n_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.LSTM_Batch_layers = []
        for _ in range(n_layers):
            LSTM_Batch_layer = ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same',
                                          return_sequences=True)
            BN = BatchNormalization()
            self.LSTM_Batch_layers.extend([LSTM_Batch_layer, BN])
        self.lamda = Lambda(lambda x: tf.reshape(x, [x.shape[0], x.shape[1], -1]))
        self.dense = Dense(num_classes, activation="softmax")
        self.pool = GlobalAveragePooling3D()
        # self.dense = Dense(num_classes)

    # def summary(self):
    #     x = Input(shape=(10, 40, 40, 1))
    #     model = tf.keras.Model(inputs=[x], outputs=self.call(x, ))
    #     return model.summary()

    def call(self, inputs):
        x = inputs
        for layer in self.LSTM_Batch_layers:
            x = layer(x)
        # x = self.flatten(x)
        # print(x.shape)
        # x = self.lamda(x)
        x = self.dense(x)
        x = self.pool(x)
        output = x
        return output


class ImuModelSequential:
    def __init__(self, n_layers, filters, num_classes):
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3), input_shape=(None, 7, 15, 6), padding='same',
                                  return_sequences=True))
        self.model.add(BatchNormalization())

        for _ in range(n_layers-1):
            self.model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same',
                                      return_sequences=True))
            self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation="softmax"))


if __name__ == "__main__":
    dummy_input = np.zeros((2, 55, 7, 15, 6))
    # imu_model = ImuModelSequential(4, 40, 10)
    imu_model = ImuModelConvLstm(4, 40, 2)
    imu_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    imu_model.summary()
    print("H")


