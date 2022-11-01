import tensorflow as tf
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense


class ImuModel(tf.keras.Model):
    def __init__(self, n_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        # LSTM_Batch_layer = ConvLSTM2D(filters=self.filters, kernel_size=(3, 3), input_shape=(None, 40, 40, 1),
        #                               padding='same', return_sequences=True)
        # BN = BatchNormalization()
        # self.LSTM_Batch_layers = [LSTM_Batch_layer, BN]
        self.LSTM_Batch_layers = []
        for _ in range(n_layers):
            LSTM_Batch_layer = ConvLSTM2D(filters=self.filters, kernel_size=(3, 3), padding='same',
                                          return_sequences=True)
            BN = BatchNormalization()
            self.LSTM_Batch_layers.extend([LSTM_Batch_layer, BN])

        self.flatten = Flatten()
        self.dense = Dense(num_classes)

    def call(self, inputs):
        x = self.LSTM_Batch_layers[0](inputs)
        for i in range(1, len(self.LSTM_Batch_layers)):
            x = self.LSTM_Batch_layers[i](x)
        x = self.flatten(x)
        output = self.dense(x)
        return output


if __name__ == "__main__":
    imu_model = ImuModel(4, 40, 10)
    _ = imu_model(tf.zeros((1, 10, 40, 40, 1)))
    imu_model.summary()
    print("H")
