import util
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    use_old = False
    dataset_dir = os.path.join("dataset", "old_dataset" if use_old else "new_dataset")
    dataset = util.UPBDataset(dataset_dir, train=True, augm=False)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    mean, std = .0, .0
    nb_samples = 0

    for data in tqdm(dataloader):
        img = data["img"]
        batch_size = img.size(0)
        img = img.view(batch_size, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std  += img.std(2).sum(0)
        nb_samples += batch_size

    mean /= nb_samples
    std /= nb_samples

    print("mean", mean)
    print("std", std)
