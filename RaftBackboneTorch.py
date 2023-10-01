#!/usr/bin/env python
"""
Provides the Raft model for the computation of optical flow from the images.

It contains all the methods required for model instantiation, image pre-processing and inference.
"""

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.2 (21/12/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

import torch
import skimage
import tensorflow as tf
from skimage.color import rgba2rgb
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Small_Weights
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small


# https://pytorch.org/vision/0.14/auto_examples/plot_optical_flow.html#sphx-glr-auto-examples-plot-optical-flow-py
# https://pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html#torchvision.models.optical_flow.raft_small
class RaftBackboneTorch:
    """
    Class that contains all the functions for the computation of Optical Flow
    """
    def __init__(self, resize_ratio=1.0, DEVICE="cuda"):
        """
        Creates the Raft model for optical flow.
        :param resize_ratio: the ratio used to scale the input images. Instead of a ratio, a 2-element array can be
        provided with the width and height of the desired resized image
        :param DEVICE: the device used for running the computations, e.g. GPU or CPU
        """
        # Model inputs
        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        # Other class properties
        self.resize_ratio = resize_ratio

        # Create model
        self.device = DEVICE
        self.model = None
        self.create_model()

    def preprocess(self, frames_batch):
        """
        Method that preprocesses the batch of images by resizing them to the desired size,
        :param frames_batch: the array of images to pre-process
        :return: two arrays, each containing one of the 2 images required for the computation of optical flow
        """
        dim = list(frames_batch.shape[2:])
        if self.resize_ratio != 1.0:  # In the case that the resize ratio is provided as an array of W x H dimensions
            if type(self.resize_ratio) != float:
                dim = self.resize_ratio
            else:  # In the case that a float number resize ratio is provided
                width = int(frames_batch.shape[3] * self.resize_ratio)
                height = int(frames_batch.shape[2] * self.resize_ratio)
                dim = [height, width]
        frames_batch = F.resize(frames_batch, size=dim)

        # Create the arrays that represent the start and end images of optical flow. They are the flight images just
        # shifted by 1
        img1_batch = frames_batch[:-1, :, :, :]
        img2_batch = frames_batch[1:, :, :, :]
        return self.transforms(img1_batch, img2_batch)

    def create_model(self):
        """
        Instantiates the raft model
        :return: None
        """
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    def delete_model_from_gpu(self):
        """
        Deletes the model from the GPU to free unused memory. This can help prevent memory leaks and ensure that there
        is a clean slate for working with other models or tasks.
        :return: None
        """
        del self.model
        self.create_model()

    def predict(self, frames, switch_return_img=False):
        """
        Method that computes the optical flow of an array of concatenated frames with overlap. In optical flow, two
        images are required in the computation, namely the Reference image (I) and the Target image (I'). With overlap
        it is meant that the target image in one optical flow computation is the reference image in the next optical
        flow computation. As a result, each image is used twice.
        :param frames: the array of frames used in the optical flow computation
        :param switch_return_img: whether the images should be returned apart from the flow
        :return: the optical flow frames and te original images
        """
        with torch.no_grad():
            # Preprocessing the images
            try:
                frames_batch = torch.permute(torch.tensor(rgba2rgb(skimage.io.imread_collection(frames))), (0, 3, 1, 2))
            except:
                frames_batch = torch.permute(torch.tensor(skimage.io.imread_collection(frames)), (0, 3, 1, 2))
            img1_batch, img2_batch = self.preprocess(frames_batch)

            # Sending the image batches to the device and computing the optical flow
            predicted_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))[-1]
            flo_lst = flow_to_image(predicted_flows).permute(0, 2, 3, 1).cpu().numpy()  # converts flow to RGB image

            # Adapting the flow and images to the right format tensors
            flo_lst = tf.stack(flo_lst)
            img_lst = []
            if switch_return_img:
                img_lst = img1_batch.permute(0, 2, 3, 1).cpu().numpy()
                img_lst = tf.stack(img_lst)

        return flo_lst, img_lst
