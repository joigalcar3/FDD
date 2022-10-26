import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import numpy as np

mpl.use('TkAgg')
figure_number = 1

# User input
flight_data_number = 43
switch_include_angle_mode = True
FFT_frequency = 1

# Configuration
data_base_folder = "D:\\AirSim_project_512_288"

# Data extraction
IMU_data_folder = os.path.join(data_base_folder, "Sensor_data")
flight_data_folder = os.path.join(data_base_folder, "Flight_info")
flight_data_file = os.path.join(flight_data_folder,
                                next(filter(lambda x: ".csv" in x and str(flight_data_number) in x,
                                            os.listdir(flight_data_folder))))
filename = "imu.csv"

# Obtaining the flight names
flights_info = pd.read_csv(flight_data_file)
flight_names = flights_info["Sensor_folder"]

# Iterating over all the flights
for i in range(len(flight_names)):
    # Retrieving flight and data information
    flight_info = flights_info.iloc[i]
    flight_name = flight_names.iloc[i]
    data_directory = os.path.join(IMU_data_folder, flight_name, filename)
    data = pd.read_csv(data_directory)

    # Obtain flight information
    flight_start_time = data['# timestamps'][0]
    failure_timestamp = (flight_info["Failure_timestamp"]-flight_start_time) * 1e-9
    damage_coefficient = flight_info["Failure_magnitude"]
    start_propeller_angle = flight_info["Start_propeller_angle"]
    failure_mode = flight_info["Failure_mode"]
    failure_bool = flight_info["Failure"]

    if not failure_bool:
        damaged_propeller = -1
    else:
        damaged_propeller = (failure_mode - 1) // 16

    if not switch_include_angle_mode:
        failure_mode = (damage_coefficient/0.2-1 + damaged_propeller * 4) + 1

    # Obtain IMU information
    d_timestamps = (data['# timestamps']-flight_start_time) * 1e-9
    slice_duration = 1/FFT_frequency
    for time_end in np.arange(slice_duration, d_timestamps.iloc[-1]+slice_duration, slice_duration):
        time_slice_index = d_timestamps[(time_end - slice_duration <= d_timestamps) & (d_timestamps < time_end)].index
        fs = len(time_slice_index)/(d_timestamps.iloc[time_slice_index[-1]]-d_timestamps.iloc[time_slice_index[0]])
        data_slice = data.iloc[time_slice_index]

        # Initialize the gyroscope and accelerometer plots
        fig_gyro, ax_gyro = plt.subplots(3, 2, num=figure_number)
        figure_number += 1
        fig_acc, ax_acc = plt.subplots(3, 2, num=figure_number)
        figure_number += 1
        for data_state_key in data.keys():
            if "timestamp" in data_state_key or "orientations" in data_state_key:
                continue
            if "angular" in data_state_key:
                fig = fig_gyro
                ax = ax_gyro
                ylabel_func = lambda xyz: f"$M_{xyz}$ [Nm]"
            else:
                fig = fig_acc
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

            # Compute the Short Time Fourier Transform
            # f, t, Zxx = signal.stft(data_state, fs, nperseg=min(len(data_state), int(fs*slice_duration)), scaling='psd')
            nperseg = int(min(len(data_state), int(fs * slice_duration))/4)
            f, t, Zxx = signal.stft(data_state, fs, nperseg=nperseg, scaling='psd', noverlap=3*nperseg//4)

            # Plot original signal
            ax1.plot(d_timestamps.iloc[time_slice_index], data_state)
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax1.axvline(failure_timestamp, color="r", linestyle="--")
            ax1.set_title(f'Signal {data_state_key}')
            ax1.set_ylabel(ylabel)
            ax1.set_xlabel('Time [sec]')
            ax1.grid(True)

            # Plot the STFT signal
            ax2.pcolormesh(t + time_end - slice_duration, f, np.abs(Zxx), vmin=0, shading="flat")
            if failure_bool and time_end > failure_timestamp > time_end - slice_duration:
                ax2.axvline(failure_timestamp, color="r", linestyle="--")
            ax2.set_title(f'STFT Magnitude for: {data_state_key}')
            ax2.set_ylabel('Frequency [Hz]')
            ax2.set_xlabel('Time [sec]')
        fig_acc.tight_layout()
        fig_gyro.tight_layout()
        plt.show()


