#!/usr/bin/env python3
"""
Runs the right model given the inputs in user_input
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
from IMU_processing.FFT.user_input import *
from IMU_processing.FFT.FlightDataGenerator import FlightDataGenerator
from IMU_processing.FFT.LstmFusionModel import LstmFusionModel
from IMU_processing.FFT.DeepNeuralFusionModel import DeepNeuralFusionModel
from IMU_processing.FFT.plotter_helper import plot_predictions, create_confusion_matrix, \
    compute_maximum_dataset_sample_timesteps
from IMU_processing.FFT.helper_func import define_callbacks, BatchLogging, convert_to_dataset

# Obtain data set
generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                   "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                   "n_time_steps_des": desired_timesteps, "raft_resize_ratio": raft_resize_ratio,
                   "switch_failure_modes": switch_failure_modes, "switch_flatten": switch_flatten,
                   "shuffle_flights": shuffle_flights, "switch_include_camera": switch_include_camera,
                   "switch_include_IMU": switch_include_IMU, "switch_single_frame": switch_single_frame,
                   "train_split": train_split, "val_split": val_split}
train_ds, val_ds, _, data_sample_shape, generators = convert_to_dataset(FlightDataGenerator, BATCH_SIZE,
                                                                        switch_shuffle_buffer, **generator_input)

# Create callbacks
callbacks, train_writer, batch_accuracy = define_callbacks(log_directory, patience, checkpoint_name)

# Create the model
if type_model == "LSTM":
    model = LstmFusionModel(n_l, n_f, n_classes).model(data_sample_shape)
elif type_model == "NN":
    # For weights preceding a ReLU function you could use the default settings of:
    #  tf.contrib.layers.variance_scaling_initializer
    #  for tanh/sigmoid activated layers "xavier" might be more appropriate:
    #  tf.contrib.layers.xavier_initializer
    # https://stackoverflow.com/questions/43489697/tensorflow-weight-initialization
    # initializer = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_in', distribution='truncated_normal')

    model = DeepNeuralFusionModel(n_l, n_f, n_classes, layers_activations=dense_act,
                                  kernel_initializer=initializer).model(data_sample_shape)
else:
    raise ValueError(f"The type of model {type_model} has not been considered.")

if train_plot == "train":
    # Encapsulate the model within the Batchlogging model for greater logging functionality
    model = BatchLogging(model, train_writer, batch_accuracy)

    # Compile, save and fit the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if switch_load_model:
        model.load_weights(checkpoint_path)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
elif train_plot == "plotCM":
    model = BatchLogging(model, None, None)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load the weights
    checkpoint_path = f"checkpoints/{checkpoint_name}"
    model.load_weights(checkpoint_path)

    # Plot the confusion matrix
    create_confusion_matrix(model, FlightDataGenerator, BATCH_SIZE, generator_input, dataset=data_type, ask_rate_it=10,
                            ignore_healthy=True, switch_single_frame=switch_single_frame)
    plot_predictions(model, FlightDataGenerator, BATCH_SIZE, generator_input)
elif train_plot == "plotDUR":
    figure_number, minimum_length, length_lst = compute_maximum_dataset_sample_timesteps(figure_number, base_folder,
                                                                                         flight_number,
                                                                                         sampling_frequency, start_time)
elif train_plot == "plotSTFT" or train_plot == "plotOF":
    stft_generator = FlightDataGenerator(base_folder, flight_number, sampling_frequency, recording_start_time=start_time,
                                         n_time_steps_des=desired_timesteps, switch_flatten=True, switch_failure_modes=True,
                                         shuffle_flights=True, switch_single_frame=switch_single_frame,
                                         switch_include_camera=False)
    if train_plot == "plotSTFT":
        figure_number = stft_generator.plot_stft(figure_number, flight_index, end_time, duration_slice,
                                                 interactive=interactive_plot_input)
    elif train_plot == "plotOF":
        stft_generator.plot_flo(flight_index, end_time, duration_slice, interactive=interactive_plot_input)
