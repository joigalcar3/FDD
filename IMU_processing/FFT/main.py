from IMU_processing.FFT.StftGenerator import *
from IMU_processing.FFT.ImuModelConvLstm import *
from IMU_processing.FFT.convert_to_dataset import convert_to_dataset
from datetime import datetime


# %% User input for the dataset
base_folder = "D:\\AirSim_project_512_288"
flight_number = 43
sampling_frequency = 10
start_time = 1.0
desired_timesteps = 55
switch_failure_modes = False
switch_flatten = True
BATCH_SIZE = 2

# %% User input for the model
n_layers = 2
filters = 10
num_classes = 2

# %% Obtain dataset
# Create the training set
generator_input = {"data_base_folder": base_folder, "flight_data_number": flight_number,
                   "STFT_frequency": sampling_frequency, "recording_start_time": start_time,
                   "n_time_steps_des": desired_timesteps, "switch_failure_modes": switch_failure_modes,
                   "switch_flatten": switch_flatten}
train_ds, val_ds, _, _ = convert_to_dataset(StftGenerator, BATCH_SIZE, **generator_input)


# %% Compile and fit the model

# model = Sequential([ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same',
#                                input_shape=(None, train_frames.shape[2], train_frames.shape[3], train_frames.shape[4]),
#                                return_sequences=True),
#                     BatchNormalization(),
#                     Lambda(lambda x: tf.reshape(x, [x.shape[0], x.shape[1], -1])),
#                     Dense(2, activation="softmax")])
#
# model = Sequential([ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same',
#                                input_shape=(None, train_frames.shape[2], train_frames.shape[3], train_frames.shape[4]),
#                                return_sequences=True),
#                     BatchNormalization(),
#                     Reshape([train_frames.shape[1], -1]),
#                     Dense(2, activation="softmax")])

imu_model = ImuModelConvLstm(n_layers, filters, num_classes)
imu_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')

# Train the model.
imu_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[tensorboard_callback, early_stop_callback])
