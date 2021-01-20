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
from util.vis import gaussian_dist, normalize


class UPBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, augm: bool = False):
        path = os.path.join(root_dir, "train_real.csv" if train else "test_real.csv")
        files = list(pd.read_csv(path)["name"])
        self.train = train
        self.augm = augm

        self.imgs = [os.path.join(root_dir, "img_real", file + ".png") for file in files]
        self.data = [os.path.join(root_dir, "data_real", file + ".pkl") for file in files]

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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        do_aug = np.random.rand() > 0.5
        do_paug = (np.random.rand() < 0.2) and self.augm

        if do_aug and self.train:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # read image & perform color augmentation
        img = pil.open(self.imgs[idx])
        img = color_aug(img)
        np_img = np.asarray(img)

        # read ground truth turning radius
        with open(self.data[idx], "rb") as fin:
            data = pkl.load(fin)
            R = data["radius"]
            course = data["rel_course"]

        # perform perspective augmentation
        if do_paug:
            np_img, R, course = PerspectiveAugmentator.augment(
                reader=self.reader, frame=np_img, R=R,
                speed=data["speed"], frame_rate=data["frame_rate"]
            )

        # process frame
        np_img = self.reader.crop_car(np_img)
        np_img = self.reader.crop_center(np_img)
        np_img = self.reader.resize_img(np_img)

        # crop even further
        np_img = Crop.crop_center(np_img, up=0.35)

        # transpose to [C, H, W] and normalize to [0, 1]
        np_img = np_img.transpose(2, 0, 1)
        np_img = normalize(np_img)
        
        # construct gaussian distribution
        # maximum  1/R = 1/5 = 0.2
        turning = 1.0 / R
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
            "speed": torch.tensor(data["speed"]).unsqueeze(0).float()

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
