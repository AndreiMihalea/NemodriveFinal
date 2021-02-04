import os
import pickle as pkl
import argparse
from util.vis import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/pose_dataset")
    args = parser.parse_args()    
    
    imgs_path = os.path.join(args.dataset_dir, "img_real")
    data_path = os.path.join(args.dataset_dir, "data_real")

    # read imgs and data files
    imgs = os.listdir(imgs_path)
    data = os.listdir(data_path)

    # sort the to make sure they correspond
    imgs = sorted(imgs)
    data = sorted(data)

    # add complete path
    imgs = sorted([os.path.join(imgs_path, x) for x in imgs])
    data = sorted([os.path.join(data_path, x) for x in data])

    for i, (img, data) in enumerate(zip(imgs, data)): 
        # read image as numpy array
        np_img = cv2.imread(img)
        
        # read all data
        with open(data, 'rb') as fin:
            dict_data = pkl.load(fin)

        # plot data
        print(dict_data)
        plot_obs_turning(np_img, 1 / dict_data['radius'], verbose=True)
        #plot_obs_turning(np_img, dict_data["rel_course"] / 100, verbose=True)
