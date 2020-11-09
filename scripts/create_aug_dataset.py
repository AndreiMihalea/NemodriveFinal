#!/usr/bin/env python
# coding: utf-8
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import util.steering as steering
import util.transformation as transformation
import pandas as pd
import pickle as pkl
import numpy as np
import math
import cv2
import argparse
import random

from tqdm import tqdm
from util.reader import Reader, JSONReader, PKLReader

# set seed
np.random.seed(0)
random.seed(0)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="path to the directory containing the raw dataset (.mov & .json)")
parser.add_argument("--train", type=str, help="path to the text file containing the training scenes")
parser.add_argument("--use_old", action="store_true", help="use the old dataset")
parser.add_argument("--verbose", action="store_true", help="display image and data")
args = parser.parse_args()


def get_steer(course, speed, dt, eps=1e-12):
    sgn = np.sign(course)
    dist = speed * dt
    R = dist / (np.deg2rad(abs(course)) + eps)
    delta, _, _ = steering.get_delta_from_radius(R)
    steer = sgn * steering.get_steer_from_delta(delta)
    return steer


def get_course(steer, speed, dt):
    dist = speed * dt
    delta = steering.get_delta_from_steer(steer)
    R = steering.get_radius_from_delta(delta)
    rad_course = dist / R
    course = np.rad2deg(rad_course)
    return course


def augment(data, translation, rotation, intersection_distance=10):
    """
    Augment a frame
    Warning!!! this augmentation may work only for turns less than 180 degrees. For bigger turns, although it
    reaches the same point, it may not follow the real car's trajectory

    :param data: [steer, velocity, delta_time]
    :param translation: ox translation, be aware that positive values mean right translation
    :param rotation: rotation angle, be aware that positive valuea mean right rotation
    :param intersection_distance: distance where the simualted car and real car will intersect
    :return: the augmented frame, steer for augmented frame
    """
    assert abs(rotation) < math.pi / 2, "The angle in absolute value must be less than Pi/2"

    steer, _, _ = data
    eps = 1e-12

    # compute wheel angle and radius of the real car
    steer = eps if abs(steer) < eps else steer
    wheel_angle = steering.get_delta_from_steer(steer + eps)
    R = steering.get_radius_from_delta(wheel_angle)

    # estimate the future position of the real car
    alpha = intersection_distance / R  # or may try velocity * delta_time / R
    P1 = np.array([R * (1 - np.cos(alpha)), R * np.sin(alpha)])

    # determine the point where the simulated car is
    P2 = np.array([translation, 0.0])

    # compute the line parameters that passes through simulated point and is
    # perpendicular to it's orientation
    d = np.zeros((3,))
    rotation = eps if abs(rotation) < eps else rotation
    d[0] = np.sin(rotation)
    d[1] = np.cos(rotation)
    d[2] = -d[0] * translation

    # we need to find the circle center (Cx, Cy) for the simulated car
    # we have the equations
    # (P11 - Cx)**2 + (P12 - Cy)**2 = (P21 - Cx)**2 + (P22 - Cy)**2
    # d0 * Cx + d1 * Cy + d2 = 0
    # to solve this, we substitute Cy with -d0/d1 * Cx - d2/d1
    a = P1[0]**2 + (P1[1] + d[2]/d[1])**2 - P2[0]**2 - (P2[1] + d[2]/d[1])**2
    b = -2 * P2[0] + 2 * d[0]/d[1] * (P2[1] + d[2]/d[1]) + 2 * P1[0] - 2 * d[0]/d[1] * (P1[1] + d[2]/d[1])
    Cx = a / b
    Cy = -d[0]/d[1] * Cx - d[2]/d[1]
    C = np.array([Cx, Cy])

    # determine the radius
    sim_R = np.linalg.norm(C - P2)
    assert np.isclose(sim_R, np.linalg.norm(C - P1)), "The points P1 and P2 are not on the same circle"

    # determine the "sign" of the radius
    # sgn = 1 if np.cross(w2, w1) >= 0 else -1
    w1 = np.array([np.sin(rotation), np.cos(rotation)])
    w2 = P1 - P2
    sgn = 1 if np.cross(w2, w1) >= 0 else -1
    sim_R = sgn * sim_R

    # determine wheel angle
    sim_delta, _, _ = steering.get_delta_from_radius(sim_R)
    sim_steer = steering.get_steer_from_delta(sim_delta)
    return sim_steer, sim_delta, sim_R, C


def pipeline(reader: Reader, img: np.array, tx: float=0.0, ry: float=0.0):
    # convention
    tx, ry = -tx, -ry

    # transform image to tensor
    img = np.asarray(img)
    height, width = img.shape[:2]
    
    K = reader.K
    K[0, :] *= width 
    K[1, :] *= height
    
    M = reader.M
    M = np.linalg.inv(M)[:3, :]
    
    # transformation object
    # after all transformation, for the old dataset we end up
    # with an image shape of (154, 256), and for the new
    # dataset with an image shape of (152, 248)
    transform = transformation.Transformation(K, M)
    output = transform.rotate_image(img, ry)
    output = transform.translate_image(output, tx)
    output = reader.crop_car(output)
    output = reader.crop_center(output)
    output = reader.resize_img(output)
    return output


def read_metadata(metadata: str, path_img: str, path_data: str, verbose: bool = False):
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
    else:
        root_dir = os.path.join(args.root_dir, metadata)
        reader = PKLReader(root_dir, "metadata.pkl", frame_rate=frame_rate)

    frame_idx = 0
    while True:
        # get next frame corresponding to current prediction
        frame, speed, rel_course = reader.get_next_image()
        if frame.size == 0:
            break

        # make conversion form relative course to steering
        dt = 1.0 / frame_rate
        steer = get_steer(rel_course, speed, dt=dt)

        # sample the translation and rotation applied
        tx, ry = 0.0, 0.0
        sgnt = 1 if np.random.rand() > 0.5 else -1
        sgnr = 1 if np.random.rand() > 0.5 else -1

        # generate random transformation
        if np.random.rand() < 0.33:
            tx = sgnt * np.random.uniform(0.25, 1.0)
            ry = sgnr * np.random.uniform(0.05, 0.1)
        else:
            if np.random.rand() < 0.5:
                tx = sgnt * np.random.uniform(0.25, 1.5)
            else:
                ry = sgnr * np.random.uniform(0.05, 0.15)
        
        # generate augmented image
        aug_img = pipeline(reader=reader, img=frame, tx=tx, ry=ry)

        # generate augmented steering command
        aug_steer, _, _, _ = augment(
            data=[steer, speed, dt],
            translation=tx,
            rotation=ry,
        )

        # convert steer to course
        aug_course = get_course(aug_steer, speed, dt)
        
        # save image and data
        scene = metadata.split('.')[0]
        frame_path = os.path.join(path_img, scene + "." + str(frame_idx).zfill(5) + ".png")
        cv2.imwrite(frame_path, aug_img)
       
        data_path = os.path.join(path_data, scene + "." + str(frame_idx).zfill(5) + ".pkl")
        with open(data_path, "wb") as fout:
            pkl.dump({"speed": speed, "rel_course": aug_course, "tx": tx, "ry": ry}, fout)

        # go to the next frame
        frame_idx += 1
        
        if verbose:
            print("Speed: %.2f, Relative Course: %.2f" % (speed, rel_course))
            print("Course: %.2f" % (aug_course,))
            print("Frame shape:", aug_img.shape)
            cv2.imshow("aug_img", aug_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # create parent directory for the dataset
    if not os.path.exists("../dataset"):
        os.makedirs("../dataset")

    # create directory depending on the dataset
    path = os.path.join("../dataset", "old_dataset" if args.use_old else "new_dataset")
    if not os.path.exists(path):
        os.makedirs(path)

    path_img = os.path.join(path, "img_aug")
    if not os.path.exists(path_img):
        os.makedirs(path_img)

    path_data = os.path.join(path, "data_aug")
    if not os.path.exists(path_data):
        os.makedirs(path_data)

    # get train scenes
    with open(args.train, "rt") as fin:
        train_scenes = fin.read()
    train_scenes = set(train_scenes.split("\n"))

    files = os.listdir(args.root_dir)
    if args.use_old:
        metadata = [file for file in files if file.endswith(".json") and file[:-5] in train_scenes]
    else:
        metadata = files.copy()

    # read metadata
    for md in tqdm(metadata):
        read_metadata(metadata=md, path_img=path_img, path_data=path_data, verbose=args.verbose)

    # read the list of images
    aug_files = os.listdir(path_img)
    aug_files = [file[:-4] for file in aug_files]

    # create csv file with the augmented images
    path_csv = os.path.join(path, "train_aug.csv")
    df = pd.DataFrame(aug_files, columns=["name"])
    df.to_csv(path_csv, index=False)

    # just a check
    df = pd.read_csv(path_csv)
    df.head()




