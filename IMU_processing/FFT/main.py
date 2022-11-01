from IMU_processing.FFT.StftGenerator import *


# %% User input
base_folder = "D:\\AirSim_project_512_288"
flight_number = 43
sampling_frequency = 10
start_time = 1.0
desired_timesteps = 55

# Create the training set
# (time, height, width, features)
# (time, features)
output_signature = (tf.TensorSpec(shape=(desired_timesteps, None, None, None), dtype=tf.float32),
                    tf.TensorSpec(shape=(desired_timesteps, 1), dtype=tf.int8))
train_ds = tf.data.Dataset.from_generator(StftGenerator(base_folder, flight_number, sampling_frequency,
                                                        recording_start_time=start_time,
                                                        n_time_steps_des=desired_timesteps, generator_type="train"),
                                          output_signature=output_signature)
# Create the validation set
val_ds = tf.data.Dataset.from_generator(StftGenerator(base_folder, flight_number, sampling_frequency,
                                                      recording_start_time=start_time,
                                                      n_time_steps_des=desired_timesteps, generator_type="val"),
                                        output_signature=output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')
