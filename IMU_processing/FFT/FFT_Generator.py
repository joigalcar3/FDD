import tensorflow as tf
import os
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# TODO: implement example plotting
# TODO: implement STFT over previous stored values


class StftGenerator:
    def __init__(self, data_base_folder, flight_data_number, STFT_frequency, n_time_steps_des=None, f_max="inf",
                 STFT_out_size=None, sensor_type="imu", sensor_features=None, recording_start_time=0,
                 switch_include_angle_mode=False, switch_failure_modes=True):
        """
        Creates generator for signal sensor data in order to feed it to TF model. The generator transforms the signal
        data into an spectrogram using a Short Time Fourier Transform (STFT)
        :param data_base_folder: base directory with the flight info and sensor data directories
        :param flight_data_number: number of the flight info file to be used
        :param STFT_frequency: frequency at which STFT will be fone on the data
        :param n_time_steps_des: the number of timesteps that should be constant for all model data samples
        :param f_max: the maximum considered frequency considered in the spectrogram
        :param STFT_out_size: the dimensions of the output STFT image
        :param sensor_type: the sensor type being analysed: imu, gps, magnetometer...
        :param sensor_features: the sensor features whose time signals will be applied the STFT
        :param recording_start_time: time at which the data starts to be used for FFT
        :param switch_include_angle_mode: whether a different propeller angle at the start of failure is considered a
        different failure mode
        :param switch_failure_modes: whether failure modes are considered or only failure has to be detected [0,1]
        """
        flight_data_folder = os.path.join(data_base_folder, "Flight_info")
        flights_info_directory = os.path.join(flight_data_folder,
                                              next(filter(lambda x: ".csv" in x and f"_{flight_data_number}_" in x,
                                                          os.listdir(flight_data_folder))))
        self.sensor_data_directory = os.path.join(data_base_folder, "Sensor_data")
        self.STFT_frequency = STFT_frequency
        self.n_time_steps_des = n_time_steps_des
        self.f_max = f_max
        self.STFT_out_size = STFT_out_size
        self.sensor_type = sensor_type
        self.sensor_features = sensor_features
        self.recording_start_time = recording_start_time
        self.switch_include_angle_mode = switch_include_angle_mode
        self.switch_failure_modes = switch_failure_modes

        self.flights_info = pd.read_csv(flights_info_directory)
        self.flight_names = self.flights_info["Sensor_folder"]

        self.nperseg = None
        self._example = None

    def __repr__(self):
        check_nan = lambda x: "Not specified" if x is None else x
        self.select_sensor_features()
        return '\n'.join([
            f'Sensor data being processed: {self.sensor_type}',
            f'Whether a different propeller angle at the start of failure is considered a different failure mode: '
            f'{self.switch_include_angle_mode}',
            f'Whether failure modes are considered or only failure has to be detected: {self.switch_failure_modes}',
            f'Frequency at which STFT is executed: {self.STFT_frequency} [Hz]',
            f'Maximum plotted STFT frequency: {check_nan(self.f_max)} [Hz]',
            f'Initial ignored seconds of every flight: {self.recording_start_time} [s]',
            f'The output has the following form: (time, height, width, features)',
            f'Number of time steps per flight/datapoint (time): {check_nan(self.n_time_steps_des)}',
            f'Dimensions STFT output image (height,width): {check_nan(self.STFT_out_size)}',
            f'Number of output training data features (features): {len(self.sensor_features)}'])

    # @property
    # def example(self):
    #     """Get and cache an example batch of `inputs, labels` for plotting."""
    #     result = getattr(self, '_example', None)
    #     if result is None:
    #         # No example batch was found, so get one from the `.train` dataset
    #         result = next(iter(self.train))
    #         # And cache it for next time
    #         self._example = result
    #     return result

    def plot_stft(self, figure_number, flight_index, time_end, slice_duration=None, interactive=True):
        if slice_duration is None:
            slice_duration = 1 / self.STFT_frequency
        filename = f"{self.sensor_type}.csv"
        self.select_sensor_features()

        # Retrieving flight sensor data information
        flight_sensor_data = self.extract_flight_sensor_data(flight_index, filename)
        flight_start_time = flight_sensor_data['# timestamps'][0]
        d_timestamps = (flight_sensor_data['# timestamps'] - flight_start_time) * 1e-9
        failure_bool = self.flights_info.iloc[flight_index]["Failure"]

        # Obtain flight information
        failure_timestamp, failure_mode = self.extract_flight_info_data(flight_index, flight_sensor_data)

        if interactive:
            print(f"Failure timestamp: {failure_timestamp}")
            print(f"Video length: {d_timestamps.iloc[-1]}")
            slice_start = float(input("Start of the stft slice in seconds."))
            slice_duration = float(input("Duration of the stft slice in seconds."))
            time_end = slice_start + slice_duration

        # Obtain the model input data and the labels
        time_slice_index = d_timestamps[
            (time_end - slice_duration <= d_timestamps) & (d_timestamps < time_end)].index
        data_slice = flight_sensor_data.iloc[time_slice_index]
        self.nperseg = (len(d_timestamps) / (d_timestamps.iloc[-1] - d_timestamps.iloc[0]) * slice_duration) // 4
        stft_stack, f, t = self.compute_slice_stft(data_slice)

        # Initialize the gyroscope and accelerometer plots
        fig_gyro, ax_gyro = plt.subplots(3, 2, num=figure_number)
        figure_number += 1
        fig_acc, ax_acc = plt.subplots(3, 2, num=figure_number)
        figure_number += 1
        counter = 0
        for data_state_key in data_slice.keys():
            if data_state_key not in self.sensor_features:
                continue
            if "angular" in data_state_key:
                ax = ax_gyro
                ylabel_func = lambda xyz: f"$M_{xyz}$ [Nm]"
            else:
                ax = ax_acc
                ylabel_func = lambda xyz: f"$F_{xyz}$ [N]"
            if "_x" in data_state_key:
                ax1 = ax[0, 0]
                ax2 = ax[0, 1]
                ylabel = ylabel_func("x")
            elif "_y" in data_state_key:
                ax1 = ax[1, 0]
                ax2 = ax[1, 1]
                ylabel = ylabel_func("y")
            elif "_z" in data_state_key:
                ax1 = ax[2, 0]
                ax2 = ax[2, 1]
                ylabel = ylabel_func("z")
            else:
                raise ValueError("The direction mentioned is not correct")

            data_state = data_slice[data_state_key]
            Zxx = stft_stack[counter]
            counter += 1

            # Plot original signal
            ax1.plot(d_timestamps.iloc[time_slice_index], data_state)
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax1.axvline(failure_timestamp, color="r", linestyle="--")
            ax1.set_title(f'Signal {data_state_key}')
            ax1.set_ylabel(ylabel)
            ax1.set_xlabel('Time [sec]')
            ax1.grid(True)

            # Plot the STFT signal
            ax2.pcolormesh(t + time_end - slice_duration, f, np.array(Zxx), vmin=0)
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax2.axvline(failure_timestamp, color="r", linestyle="--")
            ax2.set_title(f'STFT Magnitude for: {data_state_key}')
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
        fig_acc.tight_layout()
        fig_gyro.tight_layout()
        plt.show()
        return figure_number

    def compute_slice_stft(self, data_slice):
        """
        Given a slice of data, it computes its stft.
        :param data_slice: slice of data in the form of dataframe
        :return: stack of stft
        """
        stft_stack = []
        fs = len(data_slice) / ((data_slice["# timestamps"].iloc[-1] - data_slice["# timestamps"].iloc[0]) * 1e-9)
        for data_state_key in data_slice.keys():
            if data_state_key not in self.sensor_features:
                continue
            data_state = data_slice[data_state_key]
            f, t, Zxx = signal.stft(data_state, fs, nperseg=self.nperseg, scaling='psd', noverlap=3 * self.nperseg // 4)
            Zxx = np.abs(Zxx)[np.where(f <= float(self.f_max))[0]]
            stft_image = np.expand_dims(Zxx, axis=-1)
            stft_image = tf.image.convert_image_dtype(stft_image, tf.float32)
            # print(f"Image size: {stft_image.shape}. fs: {fs}. nperseg: {self.nperseg}")
            stft_stack.append(stft_image[:, :, 0])
        return stft_stack, f[np.where(f <= float(self.f_max))[0]], t

    def convert_to_stft(self, flight_sensor_data, failure_timestamp, failure_mode):
        """
        Convert flight IMU data into fft images at a self.STFT_frequency.
        :param flight_sensor_data: flight sensor data
        :param failure_timestamp: timestamp at which failure takes place
        :param failure_mode: mode of failure. It indicates the propeller and the magnitude of the failure
        :return: IMU FFT images of self.FFT_size
        """
        slice_duration = 1 / self.STFT_frequency

        flight_start_time = flight_sensor_data['# timestamps'][0]
        d_timestamps = (flight_sensor_data['# timestamps'] - flight_start_time) * 1e-9

        # Compute the number of data points used in the stft window
        self.nperseg = (len(d_timestamps) / (d_timestamps.iloc[-1] - d_timestamps.iloc[0]) * slice_duration) // 4

        # Compute the number of time steps
        n_timesteps = int((d_timestamps.iloc[-1]-self.recording_start_time)/slice_duration)
        factor_starting = 1
        if self.n_time_steps_des is not None:
            if n_timesteps < self.n_time_steps_des:
                return 0, 0, True
            factor_starting = np.random.choice(range(n_timesteps-self.n_time_steps_des + 1)) + 1

        # Run stft for every time slice
        sample_start_time = self.recording_start_time + factor_starting * slice_duration
        if self.n_time_steps_des is not None:
            sample_end_time = sample_start_time + self.n_time_steps_des * slice_duration
        else:
            sample_end_time = d_timestamps.iloc[-1] + slice_duration
        flight_stfts = []
        flight_labels = []
        for time_end in np.arange(sample_start_time, sample_end_time, slice_duration):
            if time_end > d_timestamps.iloc[-1]:
                break
            time_slice_index = d_timestamps[
                (time_end - slice_duration <= d_timestamps) & (d_timestamps < time_end)].index
            data_slice = flight_sensor_data.iloc[time_slice_index]
            stft_stack, _, _ = self.compute_slice_stft(data_slice)
            if self.STFT_out_size is not None:
                stft_stack = tf.image.resize_with_pad(stft_stack, *self.STFT_out_size)
            flight_stfts.append(stft_stack)
            if failure_timestamp <= (data_slice["# timestamps"].iloc[-1]-flight_start_time)*1e-9 and \
                    failure_timestamp != -1:
                flight_labels.append(failure_mode)
            else:
                flight_labels.append(0)
        # flight_stfts.shape => (time, features, height, width)
        flight_stfts = tf.stack(flight_stfts)
        # flight_stfts.shape => (time, height, width, features)
        flight_stfts = tf.transpose(flight_stfts, [0, 2, 3, 1])
        # flight_labels.shape => (time, features)
        flight_labels = tf.expand_dims(tf.stack(flight_labels), axis=-1)
        return flight_stfts, flight_labels, False

    def select_sensor_features(self):
        """
        Method that selects the features that will be applied the STFT
        :return:
        """
        if self.sensor_features is not None:
            return self.sensor_features

        if self.sensor_type == "imu":
            self.sensor_features = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                                    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
        else:
            raise ValueError(f"The sensor type {self.sensor_type} is not recognised.")

    def extract_flight_sensor_data(self, flight_index, filename):
        """
        Extract the sensor data
        :param flight_index: the number of the flight
        :param filename: the name of the csv where the data is stored
        :return: flight_sensor_data
        """
        flight_sensor_data_directory = os.path.join(self.sensor_data_directory,
                                                    self.flight_names.iloc[flight_index], filename)
        flight_sensor_data = pd.read_csv(flight_sensor_data_directory)
        return flight_sensor_data

    def extract_flight_info_data(self, flight_index, flight_sensor_data):
        """
        Extract the information of the flight
        :param flight_index: the number of the flight
        :param flight_sensor_data: the data of the sensor type for a single flight
        :return: flight_sensor_data, failure_timestamp, failure_mode
        """
        flight_info = self.flights_info.iloc[flight_index]
        flight_start_time = flight_sensor_data['# timestamps'][0]
        failure_timestamp = (flight_info["Failure_timestamp"] - flight_start_time) * 1e-9
        damage_coefficient = flight_info["Failure_magnitude"]
        # start_propeller_angle = flight_info["Start_propeller_angle"]
        failure_mode = flight_info["Failure_mode"]
        failure_bool = flight_info["Failure"]

        if not failure_bool:
            damaged_propeller = -1
        else:
            damaged_propeller = (failure_mode - 1) // 16

        if not self.switch_include_angle_mode:
            failure_mode = (damage_coefficient / 0.2 - 1 + damaged_propeller * 4) + 1

        if not self.switch_failure_modes:
            failure_mode = 1

        return failure_timestamp, failure_mode

    def __call__(self):
        filename = f"{self.sensor_type}.csv"
        self.select_sensor_features()

        # Iterating over all the flights
        for i in range(len(self.flight_names)):
            # Retrieving flight sensor data information
            flight_sensor_data = self.extract_flight_sensor_data(i, filename)

            # Obtain flight information
            failure_timestamp, failure_mode = self.extract_flight_info_data(i, flight_sensor_data)

            # Obtain the model input data and the labels
            stft_frames, flight_labels, skip_flag = self.convert_to_stft(flight_sensor_data, failure_timestamp,
                                                                         failure_mode)
            if skip_flag:
                continue

            yield stft_frames, flight_labels


def compute_maximum_dataset_sample_timesteps(data_base_folder, flight_data_number, STFT_frequency,
                                             recording_start_time):
    """
    Computes the minimum number of time steps among all data samples
    :param data_base_folder: location of the flight info and sensor data
    :param flight_data_number: number of the dataset
    :param STFT_frequency: frequency at which the STFT is done
    :param recording_start_time: time at which the stft starts to be done. A value higher than 0 is used in order to
    remove all initial transients
    :return:
    """
    stft_generator = StftGenerator(data_base_folder, flight_data_number, STFT_frequency,
                                   recording_start_time=recording_start_time)
    n_flights = len(stft_generator.flight_names)
    stft_generator = stft_generator()
    minimum_length = float("inf")
    length_lst = []
    for i in range(n_flights):
        frames, labels = next(stft_generator)
        length_lst.append(labels.shape[0])
        if labels.shape[0] < minimum_length:
            minimum_length = labels.shape[0]
        if i % 100 == 0:
            print(f"{i} --> current minimum = {minimum_length}")

    plt.figure(1)
    bins = sorted(set(length_lst))
    plt.hist(length_lst, bins)
    plt.title("Histogram")

    plt.figure(2)
    plt.hist(length_lst, bins, density=True, cumulative=True)
    plt.title("Cumulative histogram")
    return minimum_length, length_lst


if __name__ == "__main__":
    #%% User input
    base_folder = "D:\\AirSim_project_512_288"
    flight_number = 43
    sampling_frequency = 10
    start_time = 1.0
    desired_timesteps = 55

    #%% Plot input
    figure_number = 1
    end_time = 2
    duration_slice = 1
    flight_index = 0
    interactive_plot_input = False

    # minimum_length, length_lst = compute_maximum_dataset_sample_timesteps(base_folder, flight_number,
    #                                                                       sampling_frequency, start_time)

    stft_generator = StftGenerator(base_folder, flight_number, sampling_frequency, recording_start_time=start_time,
                                   n_time_steps_des=desired_timesteps)
    figure_number = stft_generator.plot_stft(figure_number, flight_index, end_time, duration_slice,
                                             interactive=interactive_plot_input)
    stft_generator = stft_generator()
    for i in range(5000):
        frames, labels = next(stft_generator)
        print(f"Frames: {frames.shape}. Labels: {labels.shape}")

    print("COMPARE")