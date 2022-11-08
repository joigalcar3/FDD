import time
import torch
import skimage
import tensorflow as tf
from skimage.color import rgba2rgb
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Small_Weights
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small
from IMU_processing.FFT.helper_func import split_indices


# https://pytorch.org/vision/0.14/auto_examples/plot_optical_flow.html#sphx-glr-auto-examples-plot-optical-flow-py
# https://pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html#torchvision.models.optical_flow.raft_small
class RaftBackboneTorch:
    def __init__(self, resize_ratio=0.5, number_minibatches=1, DEVICE="cuda"):  # resize_ratio=[176, 320]
        # Model inputs
        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        # Other class properties
        self.resize_ratio = resize_ratio
        self.number_minibatches = number_minibatches

        # Create model
        self.device = DEVICE
        self.model = None
        self.create_model()

    def preprocess(self, img1_batch, img2_batch):
        dim = list(img1_batch.shape[2:])
        if self.resize_ratio != 1.0:
            if type(self.resize_ratio) != float:
                dim = self.resize_ratio
            else:
                width = int(img1_batch.shape[3] * self.resize_ratio)
                height = int(img1_batch.shape[2] * self.resize_ratio)
                dim = [height, width]
        img1_batch = F.resize(img1_batch, size=dim)
        img2_batch = F.resize(img2_batch, size=dim)
        return self.transforms(img1_batch, img2_batch)

    def create_model(self):
        """
        Instantiates the raft model
        :return:
        """
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

    def delete_model_from_gpu(self):
        del self.model
        self.create_model()

    def predict(self, frame_old, frame_new, switch_return_img=False):
        # start_time = time.time()
        with torch.no_grad():
            img1_batch = torch.permute(torch.tensor(rgba2rgb(skimage.io.imread_collection(frame_old))), (0, 3, 1, 2))
            img2_batch = torch.permute(torch.tensor(rgba2rgb(skimage.io.imread_collection(frame_new))), (0, 3, 1, 2))
            img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)

            if self.number_minibatches != 1:
                # Create minibatches of indices
                splits = list(split_indices(range(img1_batch.shape[0]), min(self.number_minibatches, img1_batch.shape[0])))
                # print(f"Time for image splitting: {time.time() - start_time}")
                # start_time = time.time()

                # Compute optical flow
                flo_lst = []
                img_lst = []
                for split in splits:
                    im1, im2 = img1_batch[split], img2_batch[split]
                    predicted_flows = self.model(im1.to(self.device), im2.to(self.device))[-1]
                    # print(f"Time for minibatch raft processing: {time.time() - start_time}")
                    # start_time = time.time()

                    # Retrieve the flow array
                    flow_imgs = flow_to_image(predicted_flows).permute(0, 2, 3, 1).cpu().numpy()
                    flo_lst.extend(flow_imgs)
                    if switch_return_img:
                        img = im1.permute(0, 2, 3, 1).cpu().numpy()
                        img_lst.extend(img)
                    # print(f"Time for flow conversion: {time.time() - start_time}")
                    # start_time = time.time()
            else:
                img_lst = []
                predicted_flows = self.model(img1_batch.to(self.device), img2_batch.to(self.device))[-1]
                flo_lst = flow_to_image(predicted_flows).permute(0, 2, 3, 1).cpu().numpy()
                if switch_return_img:
                    img_lst = img1_batch.permute(0, 2, 3, 1).cpu().numpy()

        return tf.stack(flo_lst), tf.stack(img_lst)
