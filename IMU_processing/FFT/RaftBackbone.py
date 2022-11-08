from PIL import Image
import numpy as np
import argparse
import torch
import tensorflow as tf
import os
import sys
import cv2
import time
from IMU_processing.FFT.helper_func import split_indices
raft_path = list(filter(lambda x: "RAFT" in x, sys.path))[0]
sys.path.append(os.path.join(raft_path, 'core'))
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils import flow_viz

# TODO: check if it is necessary to do the translation from 2 to 3 channels


class RaftBackbone:
    def __init__(self, resize_ratio=1.0, number_minibatches=11, demo_model="models//raft-things.pth", DEVICE="cuda"):
        # Model inputs
        demo_model = os.path.join(raft_path, demo_model)
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=demo_model, help="restore checkpoint")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
        args = parser.parse_args()

        # Create model
        self.DEVICE = DEVICE
        self.args = args
        self.model = None
        self.create_model()

        # Other class properties
        self.resize_ratio = resize_ratio
        self.number_minibatches = number_minibatches

    def create_model(self):
        """
        Instantiates the raft model
        :return:
        """
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.to(self.DEVICE)
        self.model.eval()

    def delete_model_from_gpu(self):
        del self.model
        self.create_model()

    def load_image(self, imfile):
        """
        Transform an image to the required format for the RAFT model
        :param imfile: location of the file
        :return:
        """
        img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)
        if self.resize_ratio != 1.0:
            if type(self.resize_ratio) != float:
                dim = self.resize_ratio
            else:
                width = int(img.shape[1] * self.resize_ratio)
                height = int(img.shape[0] * self.resize_ratio)
                dim = (width, height)
            # Interpolation decision (compute): https://gist.github.com/georgeblck/e3e0274d725c858ba98b1c36c14e2835
            # Interpolation decision (quality): https://chadrick-kwag.net/cv2-resize-interpolation-methods/
            # CV2 use decsion: https://www.kaggle.com/code/yukia18/opencv-vs-pil-speed-comparisons-for-pytorch-user
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def load_images(self, frames_paths):
        """
        Load a list of images
        :param frames_paths: the names of the frames
        :return:
        """
        img = self.load_image(frames_paths[0])
        n_frames = len(frames_paths)
        imgs = np.zeros((n_frames, *img.shape))
        imgs[0, None] = img
        for i in range(1, n_frames):
            img = self.load_image(frames_paths[i])
            imgs[i, None] = img
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
        return imgs.to(self.DEVICE)

    def predict(self, frame_old, frame_new, switch_return_img=False):
        start_time = time.time()
        with torch.no_grad():
            images1 = self.load_images(frame_old)
            images2 = self.load_images(frame_new)
            print(f"Time for image loading: {time.time() - start_time}")
            start_time = time.time()

            # Pad the images as required
            padder = InputPadder(images1.shape)
            image1, image2 = padder.pad(images1, images2)
            print(f"Time for image padding: {time.time() - start_time}")
            start_time = time.time()

            # Create minibatches of indices
            splits = list(split_indices(range(image1.shape[0]), min(self.number_minibatches, image1.shape[0])))
            print(f"Time for image splitting: {time.time() - start_time}")
            start_time = time.time()

            # Compute optical flow
            flo_lst = []
            img_lst = []
            for split in splits:
                im1, im2 = image1[split], image2[split]
                flow_low, flow_up = self.model(im1, im2, iters=15, test_mode=True)
                print(f"Time for minibatch raft processing: {time.time() - start_time}")
                start_time = time.time()

                # Retrieve the flow array
                flo = flow_up.permute(0, 2, 3, 1).cpu().numpy()
                # flo = list(map(lambda x: flow_viz.flow_to_image(x)[:, :, [2, 1, 0]], flo))
                flo = list(map(lambda x: flow_viz.flow_to_image(x), flo))
                flo_lst.extend(flo)
                if switch_return_img:
                    img = im1.permute(0, 2, 3, 1).cpu().numpy()
                    img_lst.extend(img)
                print(f"Time for flow conversion: {time.time() - start_time}")
                start_time = time.time()

        return tf.stack(flo_lst), tf.stack(img_lst)
