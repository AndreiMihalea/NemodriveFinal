import torch
import torch.nn
import torch.nn.functional as F


import os
import argparse
import numpy as np
import random

from models.resnet import *
from util.dataset import *
from util.io import *
from util.vis import gaussian_dist
from tqdm import tqdm



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="dataset directory")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--use_pose", action="store_true", help="use pose dataset")
    parser.add_argument("--load_model", type=str, help="checkpoint name", default=None)
    parser.add_argument("--use_synth", action="store_true", help="use synthetic dataset")
    parser.add_argument("--use_baseline", action="store_true", help="evalutate baseline ")
    args = parser.parse_args()
    return args


def main(model: torch.nn.Module, test_dataloader: DataLoader):
    losses = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            # send data to device
            for key in data:
                data[key] = data[key].to(device)

            # get output
            if args.use_baseline:
                eps = 1e-16
                softmax_output = torch.tensor(gaussian_dist(200, std=10)).unsqueeze(0).float()
                log_softmax_output = torch.log(softmax_output + eps).to(device)
            else:
                with torch.no_grad():
                    turning_logits = model(data)
                    log_softmax_output = F.log_softmax(turning_logits, dim=1)
            
            kl = criterion(log_softmax_output, data["turning_pmf"])
            kl = kl.sum(dim=1).cpu().numpy()
            losses += list(kl)

            verbose = False
            if verbose:
                img = data["img"][0].cpu().numpy().transpose((1, 2, 0))
                if args.use_baseline:
                    pred = softmax_output[0].cpu().numpy()
                else:    
                    pred = F.softmax(turning_logits, dim=1)[0].cpu().numpy()
                
                gt = data["turning_pmf"][0].cpu().numpy()
                
                import matplotlib.pyplot as plti
                fig, axs = plt.subplots(2)
                print(img.shape)
                axs[0].imshow(img)
                axs[1].plot(np.arange(len(pred)), pred)
                axs[1].plot(np.arange(len(pred)), gt)
                plt.show()

    results = {
        "model": args.load_model,
        "sum": np.sum(losses),
        "mean": np.mean(losses),
        "std": np.std(losses),
        "min": np.min(losses),
        "max": np.max(losses),
        "median": np.median(losses)
    }
    return results


if __name__ == "__main__":
    # parse arguments
    args = get_args()

    # set seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    model, nbins = None, 401
    model = RESNET(no_outputs=nbins).to(device)

    # load model
    path = os.path.join("snapshots", args.load_model, "ckpts", "default.pth")
    load_ckpt(path, [('model', model)])
    model.eval()

    # define criterion
    criterion = nn.KLDivLoss(reduction="none")

    # define dataloader
    dataset_dir = os.path.join(args.dataset_dir, "pose_dataset" if args.use_pose else "gt_dataset")
    test_dataset = UPBDataset(dataset_dir, train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    results = main(model, test_dataloader)
    print(results)
    
    results_path = "./results_scale"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    path = os.path.join(results_path, args.load_model)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "open_loop.pkl"), "wb") as fout:
        pkl.dump(results, fout)
