import os
import pandas as pd
import numpy as np
import pickle as pkl
import PIL.Image as pil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from .reader import JSONReader
from .augment import PerspectiveAugmentator
from simulator.transformation import Crop
from util.vis import gaussian_dist, normalize, normalize_with_neg


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class UPBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, augm: bool = False,
            synth: bool = False, scale: float = 1.0, roi: str = 'seg_soft'):
        path = os.path.join(root_dir, "train_real.csv" if train else "test_real.csv")
        files = list(pd.read_csv(path)["name"])
        self.train = train
        self.augm = augm
        self.scale = scale
        self.roi = roi

        self.imgs = [os.path.join(root_dir, "img_real", file + ".png") for file in files]
        self.data = [os.path.join(root_dir, "data_real", file + ".pkl") for file in files]

        self.seg_labels_hard = [os.path.join(root_dir, "hard_seg_labels", file + ".npy") for file in files]
        self.seg_labels_soft = [os.path.join(root_dir, "soft_seg_labels", file + ".npy") for file in files]
        self.gt_labels_hard = [os.path.join(root_dir, "pose_hard_labels", file + ".npy") for file in files]
        self.gt_labels_soft = [os.path.join(root_dir, "pose_soft_labels", file + ".npy") for file in files]

        if self.roi == 'seg_hard':
            self.roi_data = self.seg_labels_hard
        elif self.roi == 'seg_soft':
            self.roi_data = self.seg_labels_soft
        elif self.roi == 'gt_hard':
            self.roi_data = self.gt_labels_hard
        elif self.roi == 'gt_soft':
            self.roi_data = self.gt_labels_soft

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # define reader for json that contains
        # intrinsic/extrinsic camera matrix and
        # other cropping operations
        self.reader = JSONReader()

        # buffer for synthethic dataset transformation
        self.synth_buff = []
        self.synth = synth

        # if generating synthetic dataset for testing
        if synth:
            self.synth_buff = [self.__sample() for i in range(len(self.imgs))]


    def __sample(self):
        tx, ry = 0., 0.
        sgnt = 1 if np.random.rand() > 0.5 else -1
        sgnr = 1 if np.random.rand() > 0.5 else -1

        # generate random transformation
        if np.random.rand() < 0.33:
            tx = sgnt * np.random.uniform(0.5, 1.2)
            ry = sgnr * np.random.uniform(0.05, 0.12)
        else:
            if np.random.rand() < 0.5:
                tx = sgnt * np.random.uniform(0.5, 1.5)
            else:
                ry = sgnr * np.random.uniform(0.05, 0.25)
        return tx, ry


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        # get the color augmentation and the perspective
        # augmentation flags
        do_aug = np.random.rand() > 0.5
        do_paug = (np.random.rand() < 0.5) and self.augm

        # read the image
        img = pil.open(self.imgs[idx])

        # read the roi
        if self.roi:
            roi_map_path = self.roi_data[idx]
            roi_map = np.sigmoid(np.load(roi_map_path))
        else:
            roi_map = None

        # read ground truth turning radius
        with open(self.data[idx], "rb") as fin:
            data = pkl.load(fin)
            R = data["radius"] * self.scale
            course = data["rel_course"]

        # if test and generat synthetic
        if not self.train and self.synth:
            np_img = np.asarray(img)

            # augment with a predefined tx and ry
            np_img, np_roi_map, R, course = PerspectiveAugmentator.augment(
                reader=self.reader, frame=np_img, roi_map=roi_map, R=R,
                speed=data["speed"], frame_rate=data["frame_rate"],
                transf=self.synth_buff[idx]
            )

        else:
            # color augmentation object
            if do_aug and self.train:
                color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)

            # read image & perform color augmentation
            img = color_aug(img)
            np_img = np.asarray(img)
            np_roi_map = roi_map.copy()

            # perform perspective augmentation
            if do_paug:
                np_img, np_roi_map, R, course = PerspectiveAugmentator.augment(
                    reader=self.reader, frame=np_img, roi_map=roi_map, R=R,
                    speed=data["speed"], frame_rate=data["frame_rate"]
                )

        # process frame
        np_img = self.reader.crop_car(np_img)
        np_img = self.reader.crop_center(np_img)
        np_img = self.reader.resize_img(np_img)
        # transpose to [C, H, W] and normalize to [0, 1]
        np_img = np_img.transpose(2, 0, 1)
        np_img = normalize(np_img)

        # process roi map
        np_roi_map = self.reader.crop_car(np_roi_map)
        np_roi_map = self.reader.crop_center(np_roi_map)
        np_roi_map = self.reader.resize_img(np_roi_map)
        np_roi_map = normalize_with_neg(np_roi_map)
        np_roi_map = np_roi_map[:, :, None]
        np_roi_map = np_roi_map.transpose(2, 0, 1)
        np_roi_map = np.nan_to_num(np_roi_map)

        # construct gaussian distribution
        # maximum  1/R = 1/5 = 0.2
        turning = np.clip(1.0 / R, -0.19, 0.19)
        pmf_turning = gaussian_dist(200 + 1000 * turning, std=10)

        # construct gaussina distribution
        # for course
        #course = np.clip(course, -20, 20)
        #pmf_course = gaussian_dist(200 + 10 * course)

        return {
            "img": torch.tensor(np_img).float(),
            "turning_pmf": torch.tensor(pmf_turning).float(),
            "turning": torch.tensor(turning).float(),
            #"turning_pmf": torch.tensor(pmf_course).float(),
            #"turning": torch.tensor(course).float(),
            "speed": torch.tensor(data["speed"]).unsqueeze(0).float(),
            "roi": torch.tensor(np_roi_map).float()
        }


if __name__ == "__main__":
    # define dataset
    path_dataset = os.path.join("dataset","gt_dataset")
    train_dataset = UPBDataset(path_dataset, train=True)

    for i in range(100):
        data = train_dataset[i]
        img = np.transpose(data["img"], (1, 2, 0))
        plt.imshow(img)
        plt.show()
