from PIL import Image
import numpy as np
import argparse
import torch
import os
import sys
import cv2

raft_path = list(filter(lambda x: "RAFT" in x, sys.path))[0]
sys.path.append(os.path.join(raft_path, 'core'))
from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils import flow_viz


# TODO: check if it is necessary to do the translation from 2 to 3 channels


class RaftBackbone:
    def __init__(self, resize_ratio=1.0, demo_model="models//raft-things.pth", DEVICE="cuda"):
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
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(self.DEVICE)
        self.model.eval()

        # Other class properties
        self.resize_ratio = resize_ratio

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
            # Interpolation decision: https://gist.github.com/georgeblck/e3e0274d725c858ba98b1c36c14e2835
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return img

    def load_images(self, folder_images, frames_names):
        """
        Load a list of images
        :param folder_images:
        :param frames_names:
        :return:
        """
        for frame_name in frames_names:
            imfile = os.path.join(folder_images, frame_name)
            img = self.load_image(imfile)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.DEVICE)

    def predict(self, folder_images, frame_old, frame_new, switch_return_img=False):
        with torch.no_grad():
            image1 = self.load_images(folder_images, frame_old)
            image2 = self.load_images(folder_images, frame_new)

            # Pad the images as required
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Compute optical flow
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

            # Retrieve the flow array
            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flo = flow_viz.flow_to_image(flo)
            flo = flo[:, :, [2, 1, 0]]

        if switch_return_img:
            img = image1[0].permute(1, 2, 0).cpu().numpy()
            return flo, img

        return flo, 0
