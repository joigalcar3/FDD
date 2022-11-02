import tensorflow as tf
from datetime import datetime
from keras.layers import LSTM, Dense, BatchNormalization, Input
from keras.models import Model

from IMU_processing.FFT.StftGenerator import *
from IMU_processing.FFT.convert_to_dataset import convert_to_dataset


class ImuModelConvLstmSubclassing(tf.keras.Model):
    def __init__(self, n_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.LSTM_Batch_layers = []
        for i in range(n_layers):
            LSTM_Batch_layer = LSTM(filters, return_sequences=True, name=f"LSTM_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.LSTM_Batch_layers.extend([LSTM_Batch_layer, BN])
        self.dense = Dense(num_classes, activation="softmax", name="classifier")

    def call(self, inputs):
        x = inputs
        for layer in self.LSTM_Batch_layers:
            x = layer(x)
        x = self.dense(x)
        output = x
        return output


class ImuModelConvLstmFunctional:
    def __init__(self, n_layers, filters, num_classes):
        self.LSTM_Batch_layers = []
        for i in range(n_layers):
            LSTM_Batch_layer = LSTM(filters, return_sequences=True, name=f"LSTM_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.LSTM_Batch_layers.extend([LSTM_Batch_layer, BN])
        self.dense = Dense(num_classes, activation="softmax", name="classifier")

    def model(self, input_shape):
        input_layer = Input(shape=(*input_shape,))
        x = input_layer
        for layer in self.LSTM_Batch_layers:
            x = layer(x)
        x = self.dense(x)
        output = x
        return Model(inputs=input_layer, outputs=output)


if __name__ == "__main__":
    # %% User input for the dataset
    base_folder = "D:\\AirSim_project_512_288"
    flight_number = 43
    sampling_frequency = 10
    start_time = 1.0
    desired_timesteps = 55
    switch_failure_modes = False
    switch_flatten = True
    BATCH_SIZE = 10

    # %% User input for the model
    l = 2
    f = 10
    n_classes = 2
    checkpoint_name = "saved_model"

    #%% Obtain data set
    generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                       "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                       "n_time_steps_des": desired_timesteps, "switch_failure_modes": switch_failure_modes,
                       "switch_flatten": switch_flatten}   #, "train_split": 0.20, "val_split": 0.01
    train_ds, val_ds, data_sample_shape, generators = convert_to_dataset(StftGenerator, BATCH_SIZE, **generator_input)

    # Create callbacks
    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq="batch")

    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(f"checkpoints/{checkpoint_name}", verbose=1,
                                                                   monitor="val_loss", save_best_only=True,
                                                                   save_weights_only=True)
    callbacks = [tensorboard_callback, early_stop_callback, model_checkpoint_callback]

    # %% Create and train model
    # model = ImuModelConvLstmSubclassing(l, f, n_classes)
    model = ImuModelConvLstmFunctional(l, f, n_classes).model(data_sample_shape)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    print("HOLA")
