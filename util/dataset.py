import os
import pandas as pd
import numpy as np
import pickle as pkl
import PIL.Image as pil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def gaussian_dist(mean=200.0, std=5, nbins=401):
    x = np.arange(401)
    pdf = np.exp(-0.5 * ((x - mean) / std)**2)
    pmf = pdf / pdf.sum()
    return pmf

def normalize(img):
    return img / 255.

def unnormalize(img):
    return (img * 255).astype(np.uint8)


class UPBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool=True, augm: bool=False):
        path = os.path.join(root_dir, "train_real.csv" if train else "test_real.csv")
        files = list(pd.read_csv(path)["name"])
        self.train = train

        self.imgs = [os.path.join(root_dir, "img_real", file + ".png") for file in files]
        self.data = [os.path.join(root_dir, "data_real", file + ".pkl") for file in files]

        if train and augm:
            path = os.path.join(root_dir, "train_aug.csv")
            files = list(pd.read_csv(path)["name"])
            
            self.imgs += [os.path.join(root_dir, "img_aug", file + ".png") for file in files]
            self.data += [os.path.join(root_dir, "data_aug", file + ".pkl") for file in files]


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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        do_aug = np.random.rand() > 0.5

        if do_aug and self.train:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # read image & performe color augmentation
        img = pil.open(self.imgs[idx])
        img = color_aug(img)

        # transpose to [C, H, W] and normalize to [0, 1]        
        np_img = np.asarray(img)
        np_img = np_img.transpose(2, 0, 1)
        np_img = normalize(np_img)

        # read data
        with open(self.data[idx], "rb") as fin:
            data = pkl.load(fin)


        # target
        rel_course = np.clip(data['rel_course'], -20, 20)
        pmf = gaussian_dist(mean=10 * rel_course + 200.)

        return {
            "img": torch.tensor(np_img).float(),
            "rel_course": torch.tensor(pmf).float(),
            "rel_course_val": torch.tensor(rel_course).float(),
            "speed": torch.tensor(data["speed"]).unsqueeze(0).float()
        }



if __name__ == "__main__":
    x = gaussian_dist()
    sns.lineplot(np.arange(401), x)
    plt.show()