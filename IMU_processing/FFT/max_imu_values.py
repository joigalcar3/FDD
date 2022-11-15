import os
import pandas as pd
import numpy as np


directory = "D:\\AirSim_project_512_288\\Sensor_data"
start_time = 1
files = list(filter(lambda x: "Box" not in x, os.listdir(directory)))
acc_lst = []
gyro_lst = []
exceptions_gyro = {}
exceptions_acc = {}
for i, file in enumerate(files):
    print(i)
    content = pd.read_csv(os.path.join(directory, file, "imu.csv"))
    timestamps = content["# timestamps"]
    content = content[(timestamps-timestamps[0])*1e-9 > start_time]
    acc = content[["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]]
    gyro = content[["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]]
    acc_max = np.abs(acc).max().max()
    gyro_max = np.abs(gyro).max().max()
    if np.isnan(acc_max) or np.isnan(gyro_max):
        continue
    if gyro_max > 1:
        exceptions_gyro[file] = [acc_max, gyro_max]
    if acc_max > 110:
        exceptions_acc[file] = [acc_max, gyro_max]
    acc_lst.append(acc_max)
    gyro_lst.append(gyro_max)

acc_total_max = np.max(acc_lst)
gyro_total_max = np.max(gyro_lst)
print(acc_total_max, gyro_total_max)