#!/usr/bin/env python
# coding: utf-8
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import matplotlib.pyplot as plt
import pickle as pkl
import argparse
import cv2
import glob

from tqdm import tqdm
from util.reader import Reader, JSONReader, PKLReader
from util.vis import *

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="path to the directory containing the raw datasets (.mov & .json)")
parser.add_argument("--use_old", action="store_true", help="use the old datasets")
parser.add_argument("--verbose", action="store_true", help="display image and data")
args = parser.parse_args()


def read_data(metadata: str, path_img: str, path_data: str, verbose: bool = False):
    """
    :param metadata: file containing the metadata
    :param path_img: path to the directory containing the augmented images
    :param path_data: path to the directory containing the augmented metadata
    :param verbose: verbose flag to display images while generating them
    :return: None
    """

    frame_rate = 3
    if args.use_old:
        reader = JSONReader(args.root_dir, metadata, frame_rate=frame_rate)
        scene = metadata.split(".")[0]
    else:
        root_path = os.path.join(args.root_dir, metadata)
        reader = PKLReader(root_path, "metadata.pkl", frame_rate=frame_rate)
        scene = "_".join(metadata.split("/")[-3:])

    frame_idx = 0
    while True:
        try:
            # get next frame corresponding to current prediction
            frame, speed, rel_course = reader.get_next_image()
        except:
            break

        if frame.size == 0:
            break
        
        if rel_course is None:
            continue

        # process frame
        frame = reader.crop_car(frame)
        frame = reader.crop_center(frame)
        frame = reader.resize_img(frame)

        # save image
        frame_path = os.path.join(path_img, scene + "." + str(frame_idx).zfill(5) + ".png")
        cv2.imwrite(frame_path, frame)

        # save data
        data_path = os.path.join(path_data, scene + "." + str(frame_idx).zfill(5) + ".pkl")
        with open(data_path, "wb") as fout:
            pkl.dump({"speed": speed, "rel_course": rel_course}, fout)

        # update frame count
        frame_idx += 1

        if verbose:
            print("Speed: %.2f, Relative Course: %.2f" % (speed, rel_course))
            print("Frame shape:", frame.shape)
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)


if __name__ == "__main__":
    # create parent directory
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")

    # create subdirectory for the old/new dataset
    path = os.path.join("../dataset", "old_dataset" if args.use_old else "new_dataset")
    if not os.path.exists(path):
        os.makedirs(path)

    # create the folder containing the images
    path_img = os.path.join(path, "img_real")
    if not os.path.exists(path_img):
        os.makedirs(path_img)

    # create the folder containing the data
    path_data = os.path.join(path, "data_real")
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    # read the list of scenes
    if args.use_old:
        files = os.listdir(args.root_dir)
        metadata = [file for file in files if file.endswith(".json")]
    else:
        path = os.path.join(args.root_dir, "**/*.pkl")
        files = list(glob.iglob(path, recursive=True))
        metadata = ["/".join(file.split('/')[:-1]) for file in files]

    # process all scenes
    for md in tqdm(metadata):
        read_data(metadata=md, path_img=path_img, path_data=path_data, verbose=args.verbose)

