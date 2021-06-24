import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import os
from tqdm import tqdm


if __name__ == "__main__":
    # define paths
    gt_path = "dataset/gt_dataset/"
    pose_path = "dataset/pose_dataset/"
    
    # read train with scenes
    data_names = pd.read_csv(os.path.join(gt_path, "train_real.csv"))
    data_names = list(data_names["name"])
    
    # read balancing weights
    weights = pd.read_csv(os.path.join(gt_path, "weights.csv"))
    weights = weights["name"].to_numpy()

    # define scale range to test
    lhs, rhs, points = 10, 60, 100
    scales = np.linspace(lhs, rhs, points)

    # define error buffers
    gt_curve, pose_curve = [], []

    for dn in data_names:
        scene, _ = dn.split(".")
        
        # read data
        gt_data_path = os.path.join(gt_path, "data_real", dn + ".pkl")
        pose_data_path = os.path.join(pose_path, "data_real", dn + ".pkl")

        with open(gt_data_path, "rb") as fin:
            gt_data = pkl.load(fin)

        with open(pose_data_path, "rb") as fin:
            pose_data = pkl.load(fin)

        # compute gt curvature
        gt_curve.append(1/gt_data['radius'])

        # compute pose estimator curvature
        pose_curve.append(1/pose_data['radius'])
            
    # transform to numpy
    gt_curve = np.array(gt_curve)
    pose_curve = np.array(pose_curve)

    # compute mean squared error
    MSE_unbiased, MSE = [], []
    loss = lambda x, y: (x - y)**2
    #loss = lambda x, y: np.abs(x - y)

    for scale in scales:
        l_arr = loss(
            np.clip(gt_curve, -0.2, 0.2),
            np.clip(1/scale * pose_curve, -0.2, 0.2)
        )
        
        # compute mean weighted loss and mean loss
        wl = np.sum(l_arr * weights) / np.sum(weights)
        l = np.mean(l_arr)

        # store the results
        MSE_unbiased.append(wl)
        MSE.append(l)
        
    print("Mean absolute error:")
    print(MSE)

    print("Unbiased mean absolute error:")
    print(MSE_unbiased)
    
    x_min = scales[np.argmin(MSE)]
    x_min_unbiased = scales[np.argmin(MSE_unbiased)]
    x_pred = 32.8

    x = np.concatenate([scales, scales])
    y = np.concatenate([MSE, MSE_unbiased])
    hue = np.concatenate([["MSE"] * len(scales), ["MSE_unbiased"] * len(scales)])

    print("Minimum error attenined at: %.2f" % (scales[np.argmin(MSE)]))
    print("Unbiased minimum error attenied at: %.2f" % (scales[np.argmin(MSE_unbiased)]))

    # plot errors
    ax = sns.lineplot(x, y, hue=hue)
    ax.set(xlabel="Scale factor", ylabel="Mean squared error")
    plt.axvline(x=x_min, ymin=0, ymax=1, color='b', label='Minimizer', linestyle='--')
    plt.axvline(x=x_min_unbiased, ymin=0, ymax=1, color='tab:orange', label="Unbiased minimizer", linestyle='--')
    plt.axvline(x=x_pred, ymin=0, ymax=1, color='r', label='Predicted minimizer', linestyle='--')
    plt.legend()
    plt.show()


