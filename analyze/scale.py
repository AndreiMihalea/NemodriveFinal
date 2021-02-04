import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
from tqdm import tqdm


if __name__ == "__main__":
    # define paths
    gt_path = "dataset/gt_dataset/data_real"
    pose_path = "dataset/pose_dataset/data_real"

    # read validation scenes 
    val_path = "data_split/test_scenes.txt"
    with open(val_path, "rt") as fin:
        val_scenes = fin.read()

    val_scenes = set(val_scenes.split())

    # read all recodings
    data_names = os.listdir(gt_path)
    
    # define scale range to test
    lhs, rhs, points = 10, 50, 100
    scales = np.linspace(lhs, rhs, points)

    # define error buffers
    gt_curve, pose_curve = [], []

    for dn in data_names:
        scene, _, _ = dn.split(".")
        
        # skip the scene in the training
        if scene not in val_scenes:
            continue

        # read data
        gt_data_path = os.path.join(gt_path, dn)
        pose_data_path = os.path.join(pose_path, dn)

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
    MAE = []
    loss = lambda x, y: (x - y)**2
    #loss = lambda x, y: np.abs(x - y)

    for scale in scales:
        MAE.append(np.mean(loss(
            np.clip(gt_curve, -0.2, 0.2),
            np.clip(1/scale * pose_curve, -0.2, 0.2)
        )))
        
    print("Mean absolute error:")
    print(MAE)
    
    x_min = scales[np.argmin(MAE)]
    x_pred = 32.8
    print("Minimum error attenined at: %.2f" % (scales[np.argmin(MAE)]))

    # plot errors
    ax = sns.lineplot(scales, MAE)
    ax.set(xlabel="Scale factor", ylabel="Mean absoulte error")
    plt.axvline(x=x_min, ymin=0, ymax=1, color='g', label='Minimizer')
    plt.axvline(x=x_pred, ymin=0, ymax=1, color='r', label='Predicted minimizer')
    plt.legend()
    plt.show()


