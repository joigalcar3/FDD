from keras.layers import LSTM, Dense, BatchNormalization, Input
from keras.models import Model

from IMU_processing.FFT.StftGenerator import *
from IMU_processing.FFT.helper_func import define_callbacks, BatchLogging, convert_to_dataset


class ImuModelLstmSubclassing(tf.keras.Model):
    def __init__(self, n_layers, filters, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.layers = []
        for i in range(n_layers):
            LSTM_Batch_layer = LSTM(filters, return_sequences=True, name=f"LSTM_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.layers.extend([LSTM_Batch_layer, BN])
        self.dense = Dense(num_classes, activation="softmax", name="classifier")

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.dense(x)
        output = x
        return output


class ImuModelLstmFunctional:
    def __init__(self, n_layers, filters, num_classes):
        self.layers = []
        for i in range(n_layers):
            LSTM_Batch_layer = LSTM(filters, return_sequences=True, name=f"LSTM_{i}")
            BN = BatchNormalization(name=f"BN_{i}")
            self.layers.extend([LSTM_Batch_layer, BN])
        self.dense = Dense(num_classes, activation="softmax", name="classifier")

    def model(self, input_shape):
        input_layer = Input(shape=(*input_shape,))
        x = input_layer
        for layer in self.layers:
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
    switch_failure_modes = True
    switch_flatten = True
    shuffle_flights = True
    switch_include_camera = True
    switch_shuffle_buffer = False
    # https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2
    BATCH_SIZE = 8

    # %% User input for the model
    l = 3                                         # Detection: 2
    f = 30                                        # Detection: 10
    n_classes = 17                                # Detection: 2
    checkpoint_name = f"lstm_{l}_{f}_cam{int(switch_include_camera)}_batched_multiclass_sm"     # Detection: saved_model
    epochs = 50                                   # Detection: 10   Classification: 30
    patience = 10                                 # Detection: 2   Classification: 3
    log_directory = "logs"

    #%% Obtain data set
    generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                       "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                       "n_time_steps_des": desired_timesteps, "switch_failure_modes": switch_failure_modes,
                       "switch_flatten": switch_flatten, "shuffle_flights": shuffle_flights,
                       "switch_include_camera": switch_include_camera}   #, "train_split": 0.20, "val_split": 0.01
    train_ds, val_ds, data_sample_shape, generators = convert_to_dataset(StftGenerator, BATCH_SIZE,
                                                                         switch_shuffle_buffer, **generator_input)

    # Create callbacks
    # Define the Keras TensorBoard callback.
    callbacks, train_writer, batch_accuracy = define_callbacks(log_directory, patience, checkpoint_name)

    # %% Create and train model
    # model = ImuModelLstmSubclassing(l, f, n_classes)
    model = ImuModelLstmFunctional(l, f, n_classes).model(data_sample_shape)
    model = BatchLogging(model, train_writer, batch_accuracy)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print("HOLA")
