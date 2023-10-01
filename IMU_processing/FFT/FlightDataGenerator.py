#!/usr/bin/env python
"""
Provides class FlightDataGenerator that is a generator which pre-processes each of the sensors' and camera flight data.

The generator can pre-process sensor and/or camera data. In the case that the model requires sensor data, the generator
first slices that sensor information in time slices whose size is inversely proportional to the frequency at which the
model should run and then applies the Short Time Fourier to each of those time intervals before they are being passed to
the model.

In the case that camera data is also required as input to the model, the generator also pre-processes the camera
information by extracting its optical flow. Additionally, the optical flow data is compressed by passing it through
a feature extractor; the backbone of MobileNet. Then, those compressed features are passed on to the model.

Due to the size of the dataset, it is not possible to pre-process the complete dataset and then feed it directly to the
model, so this generator progressively pulls flight sensor data, converts the sensor information to a STFT stack and the
camera data into compressed optical flow features, and feeds it to the model with the correct labels.

A STFT stack is a tensor whose channels are the stft of all the sensor signals. For instance, the first channel could
be the stft of the sensor measuremennt in the x direction, the second channel could be the sensor measurement in the
y direction, etc.
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
import os
import cv2
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from IMU_processing.FFT.RaftBackboneTorch import RaftBackboneTorch

# Limit tensorflow memory such that the pytorch model also has memory in the same machine
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

np.random.seed(0)


class FlightDataGenerator:
    """
    Generator to pre-process the signal and visual data before been fed to the machine learning model.
    """
    def __init__(self, data_base_folder, flight_data_number, STFT_frequency, n_time_steps_des=None, f_max="inf",
                 STFT_out_size=None, sensor_type="imu", camera_name="front", sensor_features=None,
                 recording_start_time=0, train_split=0.7, val_split=0.2, generator_type="train", raft_resize_ratio=0.5,
                 switch_include_angle_mode=False, switch_failure_modes=True, switch_flatten=False,
                 shuffle_flights=False, switch_single_frame=False, switch_include_camera=False,
                 switch_include_IMU=True):
        """
        Creates generator for signal and camera sensor data in order to feed it to TF model. The generator pre-processes
        the data before it is fed to the AI model.
        :param data_base_folder: base directory with the flight info and sensor data directories
        :param flight_data_number: number of the flight info file to be used
        :param STFT_frequency: frequency at which STFT will be done on the data
        :param n_time_steps_des: the number of timesteps that each data sample should be made of. It should be constant
        for all model data samples
        :param f_max: the maximum considered frequency considered in the spectrogram
        :param STFT_out_size: the dimensions of the output STFT image
        :param sensor_type: the sensor type being analysed: imu, gps, magnetometer...
        :param camera_name: the name of the camera from which the visual data is retrieved
        :param sensor_features: the sensor features to whose time signals the STFT will be applied
        :param train_split: percentage of data used for training
        :param val_split: percentage of data used for validation
        :param generator_type: whether the generator serves for training or validation data
        :param raft_resize_ratio: ratio with which the images are resized before being fed to the raft model
        :param recording_start_time: the number of initial seconds of each video and signal data that are ignored to
        avoid transients
        :param switch_include_angle_mode: whether a different propeller angle at the start of failure is considered a
        different failure mode
        :param switch_failure_modes: whether failure modes are considered for identification or only failure has to be
        detected [0,1]
        :param switch_flatten: whether the output images are flatten as a 1D vector for data fusion
        :param shuffle_flights: whether the imported flights are shuffled. If shuffled, multiple generators will not
        provide the flights in the same order
        :param switch_single_frame: whether single frames are returned instead of an array of sequential data
        :param switch_include_camera: whether camera data should be include as output from the generator
        :param switch_include_IMU: whether the IMU data should be included as output from the generator
        """
        # Obtain directories for flight data information and flight sensor data
        flight_data_folder = os.path.join(data_base_folder, "Flight_info")  # directory of the flight info files
        flights_info_directory = os.path.join(flight_data_folder,
                                              next(filter(lambda x: ".csv" in x and f"_{flight_data_number}_" in x,
                                                          os.listdir(flight_data_folder))))  # chosen file directory
        self.sensor_data_directory = os.path.join(data_base_folder, "Sensor_data")

        # Rest of instance attributes
        self.STFT_frequency = STFT_frequency
        self.slice_duration = 1 / self.STFT_frequency
        self.n_time_steps_des = n_time_steps_des
        self.f_max = f_max
        self.STFT_out_size = STFT_out_size
        self.sensor_type = sensor_type
        self.sensor_filename = f"{self.sensor_type}.csv"
        self.camera_name = camera_name
        self.sensor_features = sensor_features
        self.generator_type = generator_type
        self.recording_start_time = recording_start_time
        self.switch_include_angle_mode = switch_include_angle_mode
        self.switch_failure_modes = switch_failure_modes
        self.switch_flatten = switch_flatten
        self.shuffle_flights = shuffle_flights
        self.switch_single_frame = switch_single_frame
        self.switch_include_camera = switch_include_camera
        self.switch_include_IMU = switch_include_IMU

        # Read the flights info file in order to retrieve the name of the data files
        complete_flights_info = pd.read_csv(flights_info_directory)
        total_n_flights = len(complete_flights_info)

        # Shuffle the order of the run files.
        if self.shuffle_flights:
            complete_flights_info = complete_flights_info.sample(frac=1, random_state=1).\
                reset_index().drop("index", axis=1)

        # Create the dataset split coefficients
        if generator_type == "train":
            slice_start = 0
            slice_end = total_n_flights * train_split
        elif generator_type == "val":
            slice_start = total_n_flights * train_split
            slice_end = total_n_flights * (train_split + val_split)
        elif generator_type == 'test':
            slice_start = total_n_flights * (train_split + val_split)
            slice_end = total_n_flights
        else:
            raise ValueError("Unrecognised generator type.")

        # Slice the dataset depending if it is for training, validation or testing
        self.flights_info = complete_flights_info[int(slice_start):int(slice_end)]
        self.raw_flights_info = self.flights_info.copy()

        self.flight_names = self.flights_info["Sensor_folder"]  # data file names

        # When the camera data is also used in the FDD pipeline
        if self.switch_include_camera:
            self.raft_backbone = RaftBackboneTorch(resize_ratio=raft_resize_ratio)  # SOTA optical flow model

            # Extract one image from one flight and pass it through optical flow in order to discover the size of the
            # output
            folder_images = os.path.join(self.sensor_data_directory, self.flight_names.iloc[0], self.camera_name)
            image_name = os.path.join(folder_images, os.listdir(folder_images)[1])
            dummy_image, _ = self.raft_backbone.predict([image_name, image_name], False)
            self.raft_backbone.delete_model_from_gpu()  # delete temporal model
            torch.cuda.empty_cache()
            IMG_SHAPE = tuple(dummy_image.shape[1:])

            # Create model component that compresses the optical flow information. In this case, it is the backbone of
            # mobilenet
            # Decision between mobilenetv2 mobilenetv3large and mobilenetv3small.
            # Mobilenetv3 is more accurate and faster than mobilenetv2
            # Mobilenetv3small with alpha=1.0 could run at a frequency of 60.48 Hz
            # Mobilenetv3small with alpha=0.75 could run at a frequency of 74.82 Hz
            # Transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
            # Why mobilenets: https://keras.io/api/applications/#usage-examples-for-image-classification-models
            # Explanation: https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa
            # APIs: https://keras.io/api/applications/mobilenet/
            self.conv_feature_extractor = tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE,
                                                                                 include_top=False,
                                                                                 weights='imagenet',
                                                                                 pooling="avg",
                                                                                 alpha=0.75)
            self.conv_feature_extractor.trainable = False  # the mobilenet features are not going to be trained

        self.nperseg = None
        self._example = None

    def __repr__(self):
        """
        Method to print all the information of the generator.
        :return print string with the information of the generator
        """
        check_nan = lambda x: "Not specified" if x is None else x
        self.select_sensor_features()
        return '\n'.join([
            f'Sensor data being processed: {self.sensor_type}',
            f'Number of data points: {len(self.flight_names)}',
            f'Whether a different propeller angle at the start of failure is considered a different failure mode: '
            f'{self.switch_include_angle_mode}',
            f'Whether failure modes are considered or only failure has to be detected: {self.switch_failure_modes}',
            f'Whether the output has been flattened (heightxwidth): {self.switch_flatten}',
            f'Whether the flights have been shuffled with every epoch: {self.shuffle_flights}'
            f'Whether the generator returns single frames instead of sequences: {self.switch_single_frame}'
            f'Frequency at which STFT is executed: {self.STFT_frequency} [Hz]',
            f'Maximum plotted STFT frequency: {check_nan(self.f_max)} [Hz]',
            f'Initial ignored seconds of every flight: {self.recording_start_time} [s]',
            f'The output has the following form: (time, height, width, features)',
            f'Number of time steps per flight/datapoint (time): {check_nan(self.n_time_steps_des)}',
            f'Dimensions STFT output image (height,width): {check_nan(self.STFT_out_size)}',
            f'Number of output training data features (features): {len(self.sensor_features)}'])

    @property
    def example(self):
        """
        Get and cache an example batch of (inputs, labels) for plotting.
        :return an example output of the generator
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self()))
            # And cache it for next time
            self._example = result
        return result

    def plot_stft(self, figure_number, flight_index=0, time_end=2, slice_duration=None, interactive=True):
        """
        Function to plot the accelerations and angular velocities along the 3 axes of the accelerometer and gyroscope,
        next to their STFT transformations. The point at which a failure occurs is also highlighted.
        :param figure_number: the number of the next figure that should be plotted
        :param flight_index: the number of the flight to be plotted
        :param time_end: the last timestep used for the generation of the STFT
        :param slice_duration: the duration of each slide used for the computation of the STFT
        :param interactive: whether the plotting should be interactive, providing the user first with information
        regarding the length of the video and the timestamp of the failure
        :return: the number of the next figure
        """
        # Plotting settings
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['grid.alpha'] = 0.5
        mpl.use('TkAgg')
        font = {'size': 42,
                'family': "Arial"}
        mpl.rc('font', **font)

        # Obtain the duration of a single slice of data that should be used for the generation of the STFT.
        if slice_duration is None:
            slice_duration = 1 / self.STFT_frequency

        # Obtain the default sensor features that should be plotted
        self.select_sensor_features()

        # Retrieving flight sensor data information
        flight_sensor_data = self.extract_flight_sensor_data(flight_index)  # array with all the sensor data
        flight_start_time = flight_sensor_data['# timestamps'][0]  # time of the first measurement
        d_timestamps = (flight_sensor_data['# timestamps'] - flight_start_time) * 1e-9  # time of each measurement
        failure_bool = self.flights_info.iloc[flight_index]["Failure"]  # whether there is a failure during the flight

        # Obtain flight information
        failure_timestamp, failure_mode, _ = self.extract_flight_info_data(flight_index, flight_sensor_data)

        # In the case that the interactive mode has been activated, then the user can choose the start and end times
        if interactive:
            print(f"Failure timestamp: {failure_timestamp}")
            print(f"Video length: {d_timestamps.iloc[-1]}")
            slice_start = float(input("Start of the stft slice in seconds."))
            slice_duration = float(input("Duration of the stft slice in seconds."))
            time_end = slice_start + slice_duration

        # Obtain the data to be used for the STFT and perform the STFT
        time_slice_index = d_timestamps[
            (time_end - slice_duration <= d_timestamps) & (d_timestamps < time_end)].index
        data_slice = flight_sensor_data.iloc[time_slice_index]
        self.nperseg = (len(d_timestamps) / (d_timestamps.iloc[-1] - d_timestamps.iloc[0]) * slice_duration) // 4
        stft_stack, f, t = self.compute_slice_stft(data_slice)

        # Initialize the gyroscope and accelerometer plots
        n_rows = 3
        n_cols = 2
        fig_gyro, ax_gyro = plt.subplots(n_rows, n_cols, sharex=True, gridspec_kw={"wspace": 0.3, "hspace": 0.3},
                                         num=figure_number)
        figure_number += 1
        fig_acc, ax_acc = plt.subplots(n_rows, n_cols, sharex=True, gridspec_kw={"wspace": 0.3, "hspace": 0.3},
                                       num=figure_number)

        # Create all the plot labels
        figure_number += 1
        counter = 0
        for data_state_key in data_slice.keys():
            if data_state_key not in self.sensor_features:
                continue
            if "angular" in data_state_key:
                ax = ax_gyro
                ylabel_func = lambda xyz: f"$\Omega_{xyz}$ [rad/s]"
            else:
                ax = ax_acc
                ylabel_func = lambda xyz: f"$a_{xyz}$ [m/s$^2$]"
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

            # Extract the corresponding STFT
            data_state = data_slice[data_state_key]
            Zxx = stft_stack[counter]
            counter += 1

            # Plot original signal
            ax1.plot(d_timestamps.iloc[time_slice_index], data_state)
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax1.axvline(failure_timestamp, color="r", linestyle="--")
            # ax1.set_title(f'Signal {data_state_key}')
            ax1.set_ylabel(ylabel)
            if counter in [n_rows, n_rows*n_cols]:
                ax1.set_xlabel('Time [sec]')
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 1))
            if "angular" in data_state_key:
                ax1.yaxis.set_label_coords(-0.15, 0.5)
            else:
                ax1.yaxis.set_label_coords(-0.23, 0.5)

            # Plot the STFT signal
            ax2.pcolormesh(t + time_end - slice_duration, f, np.array(Zxx), vmin=0)
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax2.axvline(failure_timestamp, color="r", linestyle="--")
            # ax2.set_title(f'STFT Magnitude for: {data_state_key}')
            ax2.set_ylabel('f [Hz]')
            if counter in [n_rows, n_rows*n_cols]:
                ax2.set_xlabel('Time [sec]')
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 1))
            ax2.yaxis.set_label_coords(-0.15, 0.5)

        # Fine tune the plot
        fig_acc.set_size_inches(19.24, 10.55)
        fig_acc.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.13)
        fig_gyro.set_size_inches(19.24, 10.55)
        fig_gyro.subplots_adjust(left=0.125, top=0.94, right=0.98, bottom=0.13)
        plt.show()
        return figure_number

    def plot_flo(self, flight_index=0, time_end=2, plotting_interval=1, interactive=False):
        """
        Plot the optical flow and the t-x image (first image in the computation of OF)
        :param flight_index: the number of the flight
        :param time_end: the end time of the plotting interval
        :param plotting_interval: the length of the plotting interval
        :param interactive: whether the plotting should be interactive, providing the user first with information
        regarding the length of the video and the timestamp of the failure
        :return: None
        """
        # Plotting settings
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['grid.alpha'] = 0.5
        mpl.use('TkAgg')
        font = {'size': 42,
                'family': "Arial"}
        mpl.rc('font', **font)

        # Retrieving flight sensor data information
        flight_sensor_data = self.extract_flight_sensor_data(flight_index)
        _, times_dict, _ = self.extract_timestamps(flight_index, flight_sensor_data)
        failure_timestamp, _, _ = self.extract_flight_info_data(flight_index, flight_sensor_data)

        # In the case that the interactive mode has been activated, then the user can choose the start and end times
        if interactive:
            print(f"Failure timestamp: {failure_timestamp}")
            print(f"Video length: {list(times_dict.keys())[-1]}")
            time_start = float(input("Time of first OF."))
            plotting_interval = float(input(f"Time interval used for the generation of OF "
                                            f"images at {1/self.STFT_frequency} Hz."))
            time_end = time_start + plotting_interval

        # Compute the images' optical flow and the images
        flo_lst, img_lst, _ = self.compute_camera_features(time_end-plotting_interval, time_end, times_dict,
                                                           switch_return_img=True)

        # Plot the images next to each other
        for i in range(len(flo_lst)):
            flo = flo_lst[i]
            img = img_lst[i]
            img_flo = np.concatenate([img, flo], axis=0)
            cv2.imshow('image', img_flo/255.0)
            cv2.waitKey()

    def plot_flo_single(self, flight_index=0):
        """
        Plot the optical flow and the t-x image (first image in the computation of OF)
        :param flight_index: the number of the flight
        :return: None
        """
        # Retrieving flight sensor data information
        flight_sensor_data = self.extract_flight_sensor_data(flight_index)
        _, times_dict, _ = self.extract_timestamps(flight_index, flight_sensor_data)

        # Obtain image and compute optical flow. Plot both of them
        flo_lst, img_lst = self.raft_backbone.predict(list(times_dict.values()), True)
        flo = flo_lst[0]
        img_flo = np.concatenate([flo], axis=0)
        cv2.imshow('image', img_flo/255.0)
        cv2.waitKey()

    def compute_slice_stft(self, data_slice):
        """
        Given a slice of data, it computes its STFT.
        :param data_slice: slice of data in the form of dataframe
        :return: list with the STFT of all sensor features, array of sample frequencies used in the STFT and array of
        segment times
        """
        stft_stack = []
        fs = len(data_slice) / ((data_slice["# timestamps"].iloc[-1] - data_slice["# timestamps"].iloc[0]) * 1e-9)
        for data_state_key in data_slice.keys():
            if data_state_key not in self.sensor_features:
                continue
            data_state = data_slice[data_state_key]  # sensor features within the time slice
            f, t, Zxx = signal.stft(data_state, fs, nperseg=self.nperseg, scaling='psd', noverlap=3 * self.nperseg // 4)
            Zxx = np.abs(Zxx)[np.where(f <= float(self.f_max))[0]]  # filter given maximum user frequency
            stft_image = np.expand_dims(Zxx, axis=-1)  # convert to image shape
            stft_image = tf.image.convert_image_dtype(stft_image, tf.float32)  # convert STFT into TF tensor
            # print(f"Image size: {stft_image.shape}. fs: {fs}. nperseg: {self.nperseg}")
            stft_stack.append(stft_image[:, :, 0])  # remove last dimension and add tensor to list
        return stft_stack, f[np.where(f <= float(self.f_max))[0]], t

    def extract_timestamps(self, flight_index, flight_sensor_data):
        """
        Extract the timestamps of the sensors used
        :param flight_index: the number of the flight
        :param flight_sensor_data: flight sensor data
        :return: time stamps of the sensor data, time stamps of the images with their file paths and timestamp of the
        first sensor measurement
        """
        # Extract the time stamps at which the sensor measurement was taken and calibrate it with the first measurement
        # time stamp
        flight_start_time = flight_sensor_data['# timestamps'][0]
        d_timestamps = (flight_sensor_data['# timestamps'] - flight_start_time) * 1e-9

        times_dict = 0

        # In the case that the camera is included
        if self.switch_include_camera:
            folder_images = os.path.join(self.sensor_data_directory, self.flight_names.iloc[flight_index],
                                         self.camera_name)  # directory of flight images
            frames_names = sorted(os.listdir(folder_images))  # obtain image names
            frames_names = list(filter(lambda x: "png" in x, frames_names))  # remove files that are not images
            frames_times = map(lambda x: (int(x[:-4])-flight_start_time)*1e-9, frames_names)  # remove file extension
            frames_directory = map(lambda x: os.path.join(folder_images, x), frames_names)  # create full path to images
            times_dict = dict(zip(frames_times, frames_directory))  # dictionary of (image timestamps)-(image directory)

        return d_timestamps, times_dict, flight_start_time

    def compute_slice_start_end(self, failure_timestamp, d_timestamps):
        """
        Compute the start and the end of the data interval
        :param failure_timestamp: timestamp at which failure takes place
        :param d_timestamps: the timestamps of the flight used for the computation of the start and end slice times
        :return: the start and end times of the interval
        """
        # Compute the number of time steps
        n_timesteps = int((d_timestamps.iloc[-1] - self.recording_start_time) / self.slice_duration)
        if failure_timestamp != -1:
            # guarantees start before failure
            distance_failure = max(int((failure_timestamp - self.recording_start_time) / self.slice_duration), 0)
        else:
            distance_failure = float("inf")
        factor_starting = 1
        if self.n_time_steps_des is not None:
            # If the data gathered in the flight is less than the minimum required for a data sample, skip flight
            if n_timesteps < self.n_time_steps_des:
                print("SKIPPED: compute_slice_start_end")
                return 0, 0, True

            # Obtain the starting time of the interval such that it happens before the failure
            factor_starting_min = distance_failure-self.n_time_steps_des
            factor_starting_max = distance_failure
            if self.n_time_steps_des > distance_failure:
                factor_starting_min = 0
            if n_timesteps-distance_failure <= self.n_time_steps_des:
                factor_starting_max = n_timesteps-self.n_time_steps_des

            factor_starting = np.random.choice(range(factor_starting_min+1, factor_starting_max))

        # Obtain the start and end time steps
        sample_start_time = self.recording_start_time + factor_starting * self.slice_duration
        if self.n_time_steps_des is not None:
            sample_end_time = sample_start_time + self.n_time_steps_des * self.slice_duration
        else:
            sample_end_time = d_timestamps.iloc[-1] + self.slice_duration
        return sample_start_time, sample_end_time, False

    def convert_to_stft(self, flight_sensor_data, failure_timestamp, failure_mode, d_timestamps, flight_start_time,
                        sample_start_time, sample_end_time):
        """
        Convert flight IMU data into fft images at a self.STFT_frequency.
        :param flight_sensor_data: flight sensor data
        :param failure_timestamp: timestamp at which failure takes place
        :param failure_mode: mode of failure. It indicates the propeller and the magnitude of the failure
        :param d_timestamps: the flight time stamps in seconds starting at 0
        :param flight_start_time: timestamp of the first sensor measurement
        :param sample_start_time: time at which the data interval starts
        :param sample_end_time: time at which the data interval ends
        :return: tensor with all the STFT for all the timesteps for all the sensors, the label for each of the
        timesteps and the flag
        """
        # Compute the number of data points used in the STFT window
        self.nperseg = (len(d_timestamps) / (d_timestamps.iloc[-1] - d_timestamps.iloc[0]) * self.slice_duration) // 4

        flight_stfts = []  # list containing the STFT of all sensors for all time steps
        flight_labels = []
        for time_end in np.linspace(sample_start_time, sample_end_time,
                                    round((sample_end_time - sample_start_time) / self.slice_duration), endpoint=False):
            # If reached the maximum time of the sample
            if time_end > d_timestamps.iloc[-1]:
                break

            # Taking all the time stamps within the slice
            time_slice_index = d_timestamps[
                (time_end - self.slice_duration <= d_timestamps) & (d_timestamps < time_end)].index

            # Retrieving the flight sensor data for the time slice
            data_slice = flight_sensor_data.iloc[time_slice_index]
            if self.switch_include_IMU:
                stft_stack, _, _ = self.compute_slice_stft(data_slice)

                # Resize the STFT to the desired size, in the case that there is a user pre-defined size
                if self.STFT_out_size is not None:
                    stft_stack = tf.image.resize_with_pad(stft_stack, *self.STFT_out_size)
                flight_stfts.append(stft_stack)

            # In the case that the failure has already taken place, then the flight label for the slice is the failure
            # mode
            if failure_timestamp <= (data_slice["# timestamps"].iloc[-1] - flight_start_time) * 1e-9 and \
                    failure_timestamp != -1:
                flight_labels.append(failure_mode)
            else:  # otherwise, the label is 0
                flight_labels.append(0)

        # In the case that the STFT was unsuccessful or there is a corrupt failure mode, then skip the flight
        if np.sum([np.isnan(flight_stfts), np.isinf(flight_stfts)]) or \
                np.sum([np.isnan(flight_labels), np.isinf(flight_labels)]):
            print("SKIPPED: convert_to_stft")
            return 0, 0, True

        if self.switch_include_IMU:
            # flight_stfts.shape => (time, features, height, width)
            flight_stfts = tf.stack(flight_stfts)
            # flight_stfts.shape => (time, height, width, features)
            flight_stfts = tf.transpose(flight_stfts, [0, 2, 3, 1])
        # flight_labels.shape => (time, features)
        flight_labels = tf.cast(tf.expand_dims(tf.stack(flight_labels), axis=-1), tf.int8)
        return flight_stfts, flight_labels, False

    def select_sensor_features(self):
        """
        Method that selects the sensor features to which the STFT will be applied.
        :return: None
        """
        if self.sensor_type == "imu" and self.sensor_features is None:
            self.sensor_features = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                                    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
        else:
            raise ValueError(f"The sensor type {self.sensor_type} is not recognised.")

    def extract_flight_sensor_data(self, flight_index):
        """
        Extract the sensor data (e.g. imu)
        :param flight_index: the number of the flight
        :return: the flight sensor data
        """
        flight_sensor_data_directory = os.path.join(self.sensor_data_directory,
                                                    self.flight_names.iloc[flight_index], self.sensor_filename)
        flight_sensor_data = pd.read_csv(flight_sensor_data_directory)
        return flight_sensor_data

    def extract_flight_info_data(self, flight_index, flight_sensor_data):
        """
        Extract the information of the flight
        :param flight_index: the number of the flight
        :param flight_sensor_data: the data of the sensor type for a single flight
        :return: the time at which the failure took place and the failure mode
        """
        # Extracting information from the flights file for the particular flight
        flight_info = self.flights_info.iloc[flight_index]
        failure_bool = flight_info["Failure"]  # whether there was a failure
        raw_failure_timestamp = flight_info["Failure_timestamp"]  # the failure timestamp
        if np.isnan(raw_failure_timestamp):  # this particular flight has a data error: ignore flight
            print("SKIPPED: extract_flight_info_data")
            return 0, 0, True

        # In the case that it failed, extract more features
        if failure_bool:
            flight_start_time = flight_sensor_data['# timestamps'][0]  # Time at which the flight started
            failure_timestamp = (raw_failure_timestamp - flight_start_time) * 1e-9  # Time passed from start to failure
            failure_mode = flight_info["Failure_mode"]  # Number mode of failure

            # Number of the propeller that failed. Ordered clockwise from the front left propeller
            damaged_propeller = (failure_mode - 2) // 16

            # In the case that the initial angle of rotation should not affect the mode, as it is the case by default.
            if not self.switch_include_angle_mode:
                damage_coefficient = flight_info["Failure_magnitude"]  # the magnitude of the failure [0.2,0.4,...,0.8]
                failure_mode = (damage_coefficient / 0.2 - 1 + damaged_propeller * 4) + 1  # compute new mode value

            # In the case that detection only (no identification) is required. Then any failure has a failure mode of 1.
            if not self.switch_failure_modes:
                failure_mode = 1
        else:
            failure_timestamp = -1
            failure_mode = 0

        return failure_timestamp, failure_mode, False

    def compute_camera_features(self, sample_start_time, sample_end_time, times_dict, switch_return_img=False):
        """
        Compute the features obtained from the camera
        :param sample_start_time: first time at which the flight data will be taken
        :param sample_end_time: last time at which the flight data will be taken
        :param times_dict: dictionary matching the timestamps with the images' names
        :param switch_return_img: whether the images should be returned apart from the flow
        :return: the optical flow and images captured during the complete flight
        """
        # Obtaining the time stamp of the images
        d_timestamps = pd.DataFrame(list(times_dict.keys()), columns=["t"])["t"]

        # Obtain the last image before the start of the first time slice
        frame_old = times_dict[d_timestamps[d_timestamps < sample_start_time-self.slice_duration].iloc[-1]]
        frame_lst = [frame_old]

        # Iterate over all the time slices
        for time_end in np.linspace(sample_start_time, sample_end_time,
                                    round((sample_end_time - sample_start_time) / self.slice_duration), endpoint=False):
            # Extract the frames within the time slice
            frames_index = d_timestamps[(time_end - self.slice_duration <= d_timestamps) & (d_timestamps < time_end)].index

            # When there are no frames in the time slice, skip the flight
            if len(frames_index) == 0:
                # Example of flight that requires this: 20220802-023533_1. Between frames 205 and 206 there is a time
                # difference of 0.3+
                print("SKIPPED: compute_camera_features")
                return 0, 0, True

            # Obtaining the last frame of the time slice
            frame_new = times_dict[d_timestamps.iloc[frames_index[-1]]]
            frame_lst.append(frame_new)

        # Obtain the tensors with the optical flow and images captured during the complete flight
        flo_lst, img_lst = self.raft_backbone.predict(frame_lst, switch_return_img)

        return flo_lst, img_lst, False

    def __call__(self):
        """
        Method called when the generator is called as a iterator. Once the main for-loop that generates the output is
        exhausted, the call method is called from the beginning
        :return: the features from the sensors (camera and e.g. IMU) and the labels
        """
        # Select what features from the sensors should be analysed
        self.select_sensor_features()

        # Every time the complete dataset has been passed as output of the generator, the dataset is reshuffled.
        if self.shuffle_flights:
            self.flights_info = self.raw_flights_info.sample(frac=1).reset_index().drop("index", axis=1)
            self.flight_names = self.flights_info["Sensor_folder"]

        # Iterating over all the flights
        for i in range(len(self.flight_names)):
            # Retrieving flight sensor data information
            flight_sensor_data = self.extract_flight_sensor_data(i)

            # Obtain flight information
            failure_timestamp, failure_mode, skip_flag = self.extract_flight_info_data(i, flight_sensor_data)

            # In the case that the data is incorrect or corrupted
            if skip_flag:
                print(f"Flight name: {self.flight_names.iloc[i]}")
                continue

            # Obtain the timestamps from the sensors
            d_timestamps, times_dict, flight_start_time = self.extract_timestamps(i, flight_sensor_data)

            # Obtain the interval start and end times. The timestamps from the camara have priority
            if self.switch_include_camera:
                selected_timestamps = pd.DataFrame(list(times_dict.keys()), columns=["t"])["t"]
            elif self.switch_include_IMU:
                selected_timestamps = d_timestamps
            else:
                raise ValueError("The IMU and/or the camera need to be switched on.")
            sample_start_time, sample_end_time, skip_flag = \
                self.compute_slice_start_end(failure_timestamp, selected_timestamps)

            # If the data gathered in the flight is less than the minimum required for a data sample, skip flight
            if skip_flag:
                print(f"Flight name: {self.flight_names.iloc[i]}")
                continue

            # Obtain the IMU features
            stft_frames, flight_labels, skip_flag = self.convert_to_stft(flight_sensor_data, failure_timestamp,
                                                                         failure_mode, d_timestamps, flight_start_time,
                                                                         sample_start_time, sample_end_time)

            # If the STFT or the failure mode of the flight are corrupted, then skip flight
            if skip_flag:
                print(f"Flight name: {self.flight_names.iloc[i]}")
                continue

            # If the data is flattened for the fusion
            if self.switch_flatten and self.switch_include_IMU:
                stft_frames = tf.reshape(stft_frames, [stft_frames.shape[0], -1])

            output_features = stft_frames

            # Obtain camera information
            if self.switch_include_camera:
                flo_lst, _, skip_flag = self.compute_camera_features(sample_start_time, sample_end_time, times_dict)

                # When there are no frames in the time slice, skip the flight
                if skip_flag:
                    print(f"Flight name: {self.flight_names[i]}")
                    continue

                # Compress the optical flow by passing it through a feature extractor
                flo_features = self.conv_feature_extractor(flo_lst)

                # Combine the camera and sensor data into a single tensor
                if self.switch_include_IMU:
                    output_features = tf.concat([output_features, flo_features], 1)
                else:
                    output_features = flo_features

            # In the case that it is desired to return a time step at a time instead of complete flights. This is needed
            # when the data is not analysed as temporal data, e.g. fully connected NN instead of LSTM.
            if self.switch_single_frame:
                n_frames = output_features.shape[0]
                for i in range(n_frames):
                    output_feature = output_features[i]
                    flight_label = flight_labels[i]
                    yield output_feature, flight_label
            else:  # Provide the complete stream of features and labels as output
                yield output_features, flight_labels
