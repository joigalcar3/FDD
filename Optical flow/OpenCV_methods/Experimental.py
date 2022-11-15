import os
import cv2
import numpy as np
import glob
import time
from skimage.io import imread
from skimage.color import rgba2rgb


def create_video(img_folder, fps_rate_in=30, fps_rate_out=30, start_frame=0, out_folder=None, out_file_name=None):
    img_names = sorted(os.listdir(img_folder))
    image_path = os.path.join(img_folder, img_names[0])
    image = imread(image_path)
    frameSize = image.shape[1::-1]
    if out_file_name is None:
        out_file_name = img_folder.split("\\")[-1] + f"_s{start_frame}_f{fps_rate_out}"
    if out_folder is None:
        cwd = os.path.dirname(os.getcwd())
        out_folder = f'{cwd}\\video_storage\\' + img_folder.split("\\")[-1]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file_address = os.path.join(out_folder, f'{out_file_name}.avi')
    if not os.path.exists(file_address):
        out = cv2.VideoWriter(file_address, cv2.VideoWriter_fourcc(*'DIVX'), fps_rate_out, frameSize)

        for filename in img_names[start_frame::int(fps_rate_in/fps_rate_out)]:
            image_path = os.path.join(img_folder, filename)
            img = cv2.imread(image_path)
            out.write(img)

        out.release()
    return out_folder, out_file_name, file_address


def lucas_kanade_method(video_path, image_step=1, fps_rate=30, max_corners=100):
    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create random colors
    color = np.random.randint(0, 255, (max_corners, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Store video
    frameSize = old_frame.shape[1::-1]
    filename = video_path[:-4] + f"_k{image_step}_LK.avi"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps_rate, frameSize)

    # Create buffer
    buffer = [old_gray]
    buffer_p0 = [p0]
    for i in range(image_step-1):
        ret, old_frame = cap.read()
        if not ret:
            break
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            buffer[-1], old_frame, buffer_p0[-1], None, **lk_params
        )
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        good_new = p1[st == 1]
        p0 = good_new.reshape(-1, 1, 2)
        buffer_p0.append(p0.copy())
        buffer.append(old_frame)
        buffer_p0.append(p0)

    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p0 = buffer_p0.pop(0)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            buffer.pop(0), frame_gray, p0, None, **lk_params
        )

        buffer.append(frame_gray.copy())
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        # Display the demo
        img = cv2.add(frame, mask)
        out.write(img)
        cv2.imshow("frame", img)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

        # Update the previous frame and previous points
        p0 = good_new.reshape(-1, 1, 2)
        buffer_p0.append(p0.copy())
    out.release()


def dense_optical_flow(dOF_alg, video_path, skip_frame=0, fps_rate=30, image_step=1, to_gray=False):
    # Read the video and first frame
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Method selection
    if dOF_alg == 'farneback':
        method = cv2.calcOpticalFlowFarneback
        # [pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags]
        params = [0.5, 3, 15, 3, 5, 1.2, 0]   # cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        # params = [0.5, 3, 30, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
    else:
        raise ValueError(f"The dense Optical Flow method {dOF_alg} is not recognised.")

    # Store video
    frameSize = old_frame.shape[1::-1]
    filename = video_path[:-4] + f"_k{image_step}_{dOF_alg}.avi"
    # filename = video_path[:-4] + f"_k{image_step}_{'_'.join(map(str, params))}.avi"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps_rate, frameSize)
    buffer = [old_frame]
    for i in range(image_step-1):
        ret, old_frame = cap.read()
        if not ret:
            break
        if to_gray:
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        buffer.append(old_frame.copy())
    skip_frame_counter = 0
    time_lst = []
    counter = 0
    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        buffer.append(new_frame.copy())
        if skip_frame_counter == skip_frame and skip_frame != 0:
            buffer.pop(0)
            skip_frame_counter = 0
            continue
        start = time.time()
        flow = method(buffer.pop(0), new_frame, None, *params)
        end = time.time()
        elapsed_time = end - start
        time_lst.append(elapsed_time)
        print(f"Elapsed time of iter {counter}: {elapsed_time}")
        counter += 1

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use Hue and Value to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bgr)
        skip_frame_counter += 1
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
    print(f"Average elapsed time: {np.mean(time_lst)}")
    out.release()


if __name__ == "__main__":
    # User input
    # Build a list of image pairs to process
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_256_144"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_city_512_288"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576"
    img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576_2"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Sintel_clean_ambush"
    # img_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\KITTI_2015"
    fps_in = 2
    fps_out = 2
    start_frame = 0
    max_corners = 200
    step = 1
    skip_frame = 0
    if "Coen" in img_folder:
        fps_in = 30
        fps_out = 30
        start_frame = 55
        max_corners = 200
        step = 1
    elif "KITTI" in img_folder:
        fps_in = 1
        fps_out = 1
        start_frame = 0
        max_corners = 200
        step = 1
        skip_frame = 1

    # Dense OF user input
    switch_dense = True
    dOF_algorithm = 'farneback'

    # Create video from frames
    out_folder, out_file_name, file_address = create_video(img_folder, fps_in, fps_out, start_frame)
    if switch_dense:
        frames = dense_optical_flow(dOF_algorithm, file_address, fps_rate=fps_out, image_step=step, to_gray=True,
                                    skip_frame=skip_frame)
    else:
        lucas_kanade_method(file_address, step, fps_out, max_corners)
