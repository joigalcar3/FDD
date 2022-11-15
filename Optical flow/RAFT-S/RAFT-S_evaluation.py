from skimage.color import rgba2rgb
import numpy as np
import skimage
import torch
import time
import glob
import cv2
import os

from IMU_processing.FFT.RaftBackboneTorch import RaftBackboneTorch
from torchvision.utils import flow_to_image


def RAFT_S_video_prediction(image_folder, video_path, fps_out, img_start, img_end, img_step, step_viz=False):
    raft_backbone = RaftBackboneTorch(DEVICE="cpu", resize_ratio=1)

    with torch.no_grad():
        images = glob.glob(os.path.join(image_folder, '*.png')) + \
                 glob.glob(os.path.join(image_folder, '*.jpg'))

        images = sorted(images)[img_start:img_end]
        frames_batch = torch.permute(torch.tensor(rgba2rgb(skimage.io.imread_collection(images))), (0, 3, 1, 2))
        img1_batch, img2_batch = raft_backbone.preprocess(frames_batch)

        frameSize = tuple(img1_batch.shape[3:1:-1])
        filename = os.path.join(video_path,
                                video_path.split("\\")[-1] + f"_RAFT-S_s{img_start}_f{fps_out}_k{img_step}_with.avi")
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps_out, frameSize)
        time_lst = []
        counter = 0
        for imfile1, imfile2 in zip(img1_batch[:-img_step], img2_batch[img_step:]):
            start = time.time()
            predicted_flows = raft_backbone.model(torch.unsqueeze(imfile1, 0).to(raft_backbone.device),
                                                  torch.unsqueeze(imfile2, 0).to(raft_backbone.device))[-1]
            end = time.time()
            elapsed_time = end-start
            time_lst.append(elapsed_time)
            print(f"Elapsed time of iter {counter}: {elapsed_time}")
            flo_lst = flow_to_image(predicted_flows).permute(0, 2, 3, 1).cpu().numpy()[0]
            out.write(flo_lst[:, :, [2, 1, 0]])
            if step_viz:
                img = frames_batch[counter].permute(1, 2, 0).cpu().numpy()
                flo = flo_lst / 255.0
                img_flo = np.concatenate([img, flo], axis=0)
                cv2.imshow('image', img_flo[:, :, [2, 1, 0]])
                cv2.waitKey()
            counter += 1
        print(f"Average elapsed time: {np.mean(time_lst)}")
        out.release()

# %% User input for the dataset
demo_folder = "D:\\AirSim simulator\\FDD\\Optical flow\\example_images\\Coen_City_1024_576_2"
video_storage_folder = os.path.join("D:\\AirSim simulator\\FDD\\Optical flow\\video_storage",
                                    demo_folder.split("\\")[-1])
fps_out = 30
img_start = 0
img_end = -1
img_step = 3

# %% Model
RAFT_S_video_prediction(demo_folder, video_storage_folder, fps_out, img_start, img_end, img_step, step_viz=False)
