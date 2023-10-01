#!/usr/bin/env python3
"""
Contains the inputs to run all the models
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
from keras.layers import LeakyReLU, Activation
from IMU_processing.FFT.helper_func import gelu
from keras.utils.generic_utils import get_custom_objects

# User input for both models
train_plot = "plotCM"  # run training ("train"), "plotCM", plot the flights' duration ("plotDUR") "plotSTFT" or "plotOF"
type_model = "LSTM"  # what kind of model should run. "LSTM": ImuModelLSTM. "NN": DeepNeuralFusionModel
base_folder = "D:\\Coen SSD copy\\AirSim_project"  # base directory with the flight info and sensor data directories
checkpoint_path = "IMU_processing\\FFT\\checkpoints\\lstm_3_30_cam1_IMU1_batched_multiclass_sm20221122-162949"
flight_number = 19  # number of the flight info file to be used
sampling_frequency = 10  # the frequency at which the FDD pipeline runs
start_time = 1.0  # number of initial seconds of ignored video and signal data to avoid transients
desired_timesteps = 55  # the number of seconds of video that each data sample is going to be made of
switch_failure_modes = True  # whether failure identification is activated besides failure detection
switch_flatten = True  # whether the output images are flatten as a 1D vector for data fusion
shuffle_flights = True  # whether the imported flights are shuffled such that each generator is different
raft_resize_ratio = 1  # ratio with which the images are resized before being fed to the raft model
switch_include_camera = True  # whether the model should include the camara data
switch_include_IMU = True  # whether the model should include the IMU pipeline
switch_load_model = False  # whether a model checkpoint should be loaded or the model should be trained from scratch

train_split = 0.7  # percentage of the samples used for training
val_split = 0.2  # percentage of the samples used for validation

epochs = 50  # number of epochs
patience = 10  # number of epochs to wait without improving the loss function before early stopping
log_directory = "logs"  # name of the folder where model and batch metrics are stored

# Specialised inputs
if type_model == "NN":
    # General Deep NN model inputs
    switch_single_frame = True  # whether the generator should output a single time step of data at a time
    switch_shuffle_buffer = True  # whether a pre-loaded buffer of data should be created, shuffled and pre-fetched
    # https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2
    BATCH_SIZE = 1024

    # Creation of additional optional activation functions
    # Add gelu and leaky-relu so we can use it as a string
    # Add to objects: https://stackoverflow.com/questions/65302438/add-custom-activation-function-to-be-used-with-a-string
    # Gelu: https://arxiv.org/abs/1606.08415
    get_custom_objects().update({'gelu': Activation(gelu)})
    get_custom_objects().update({'leaky-relu': LeakyReLU(alpha=0.1)})

    # Deep NN architecture and training inputs
    n_l = 3                  # number of layers
    n_f = 128                # number of filters
    dense_act = "tanh"     # activation functions used in the neurons
    n_classes = 17        # number of classes for the classification problem
    initializer = "glorot_uniform"  # Initialization method of the weight
    checkpoint_name = f"dense_{dense_act}_glorot_{n_l}_{n_f}_cam{int(switch_include_camera)}_" \
                      f"IMU{int(switch_include_IMU)}_batched_multiclass_sm"
elif type_model == "LSTM":
    # General LSTM model inputs
    switch_single_frame = False  # whether the generator should output a single time step of data at a time
    switch_shuffle_buffer = False  # whether a pre-loaded buffer of data should be created, shuffled and pre-fetched
    # https://datascience.stackexchange.com/questions/20179/what-is-the-advantage-of-keeping-batch-size-a-power-of-2
    BATCH_SIZE = 64

    # LSTM architecture and training inputs
    n_l = 3  # number of layers
    n_f = 30  # number of filters
    n_classes = 17  # number of classes for the classification problem
    checkpoint_name = f"lstm_{n_l}_{n_f}_cam{int(switch_include_camera)}_" \
                      f"IMU{int(switch_include_IMU)}_batched_multiclass_sm"
else:
    raise ValueError(f"The type of model {type_model} has not been considered.")

# Inputs for plotting
data_type = "train"
if train_plot == "plotCM":
    train_split = 1  # percentage of the samples used for training
    val_split = 0.0  # percentage of the samples used for validation
    BATCH_SIZE = 1
    checkpoint_name = "lstm_3_30_cam1_IMU1_batched_multiclass_sm20221122-162949"
elif train_plot == "plotDUR" or train_plot == "plotSTFT" or train_plot == "plotOF":
    figure_number = 1
    end_time = 2
    duration_slice = 1
    flight_index = 1
    interactive_plot_input = True
