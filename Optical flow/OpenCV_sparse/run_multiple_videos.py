import cv2
import os


def compare_videos(window_titles, names):
    # window_titles = ['first', 'second', 'third', 'fourth']

    cap = [cv2.VideoCapture(i) for i in names]

    frames = [None] * len(names)
    gray = [None] * len(names)
    ret = [None] * len(names)
    frame_counter = 0
    while True:
        # If the last frame is reached, reset the capture and the frame_counter
        frame_counter += 1
        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()

        if frame_counter > cap[1].get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0  # Or whatever as long as it is the same as next line
            for i, c in enumerate(cap):
                c.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for i, f in enumerate(frames):
            if ret[i] is True:
                gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_titles[i], gray[i])

        if cv2.waitKey(250) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    folder_videos = "D:\\AirSim simulator\\FDD\\Optical flow\\OpenCV_sparse\\video_storage"
    current_scenario = "Coen_city_256_144"
    name1 = "Coen_city_256_144_PWC_s30_f30_k3_255"
    name2 = "Coen_city_256_144_RAFT_s30_f30_k3"
    name3 = "Coen_city_256_144_s30_k1"
    # name4 = "Coen_City_1024_576_2_s30_k10.5_3_15_6_5_1.2_0"
    # name5 = "Coen_City_1024_576_2_s30_k10.5_3_15_3_7_1.5_0"
    # name6 = "Coen_City_1024_576_2_s30_k10.5_3_15_3_5_1.2_1"
    # name7 = "Coen_City_1024_576_2_s30_k10.5_3_5_3_5_1.2_0"
    window_titles = [name1, name2, name3]
    names = [os.path.join(folder_videos, current_scenario, i + ".avi") for i in window_titles]
    compare_videos(window_titles, names)
