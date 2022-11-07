import tensorflow as tf
import numpy as np
from keras.layers import Dense, BatchNormalization, LeakyReLU, Activation
from keras.utils.generic_utils import get_custom_objects

from IMU_processing.FFT.ImuModelLSTM import ImuModelLstmFunctional
from IMU_processing.FFT.StftGenerator import StftGenerator
from IMU_processing.FFT.helper_func import define_callbacks, BatchLogging, convert_to_dataset
# TODO: try different activations functions --> leaky_rely, gelu
# TODO: try another kernel_initialization


# Add the GELU function to Keras
def gelu(x):
    """
    Gaussian Error Linear Units
    :param x: input to activation function
    :return:
    """
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


# Add gelu and leaky-relu so we can use it as a string
# Add to objects: https://stackoverflow.com/questions/65302438/add-custom-activation-function-to-be-used-with-a-string
# Gelu: https://arxiv.org/abs/1606.08415
get_custom_objects().update({'gelu': Activation(gelu)})
get_custom_objects().update({'leaky-relu': LeakyReLU(alpha=0.1)})


class ImuModelDeepFunctional(ImuModelLstmFunctional):
    def __init__(self, n_layers, filters, num_classes, layers_activations="tanh", kernel_initializer="glorot_uniform"):
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
    switch_single_frame = True
    BATCH_SIZE = 64

    # %% User input for the model
    l = 3                                         # Detection: 2
    f = 128                                        # Detection: 10
    dense_act = "leaky-relu"
    n_classes = 17                                # Detection: 2
    checkpoint_name = f"dense_{dense_act}_glorot_{l}_{f}_batched_multiclass_sm"     # Detection: saved_model
    epochs = 50                                   # Detection: 10   Classification: 30
    patience = 10                                 # Detection: 2   Classification: 3
    log_directory = "logs"

    # For weights preceding a ReLU function you could use the default settings of:
    #  tf.contrib.layers.variance_scaling_initializer
    #  for tanh/sigmoid activated layers "xavier" might be more appropriate:
    #  tf.contrib.layers.xavier_initializer
    # https://stackoverflow.com/questions/43489697/tensorflow-weight-initialization
    # initializer = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='truncated_normal')
    initializer = "glorot_uniform"

    #%% Obtain data set
    generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                       "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                       "n_time_steps_des": desired_timesteps, "switch_failure_modes": switch_failure_modes,
                       "switch_flatten": switch_flatten, "shuffle_flights": shuffle_flights,
                       "switch_single_frame": switch_single_frame}   #, "train_split": 0.20, "val_split": 0.01
    train_ds, val_ds, data_sample_shape, generators = convert_to_dataset(StftGenerator, BATCH_SIZE, **generator_input)

    # Create callbacks
    # Define the Keras TensorBoard callback.
    callbacks, train_writer, batch_accuracy = define_callbacks(log_directory, patience, checkpoint_name)

    # %% Create and train model
    # model = ImuModelConvLstmSubclassing(l, f, n_classes)
    model = ImuModelDeepFunctional(l, f, n_classes, layers_activations=dense_act,
                                   kernel_initializer=initializer).model(data_sample_shape)
    model = BatchLogging(model, train_writer, batch_accuracy)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print("HOLA")
