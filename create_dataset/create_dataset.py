import argparse
from tqdm import tqdm

import os
import pickle as pkl
from util.vis import *
from util.reader import JSONReader
from simulator import steering
from pose.pose_estimation import PoseEstimation


def parse_args():
    """
    Parse console arguments

    Returns
    -------
    Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="path to the directory containing the raw datasets (.mov & .json)")
    parser.add_argument("--use_pose", action="store_true", help="compute the steering from pose estimation")
    parser.add_argument("--frame_rate", type=int, help="sampling frame rate")
    parser.add_argument("--verbose", action="store_true", help="display image and data")
    args = parser.parse_args()
    return args


def read_data(metadata: str, path_img: str, path_data: str,
              frame_rate: int=3, verbose: bool = False):
    """
    Reads and save data from a video and coresponding json

    Parameters
    ----------
    metadata
        file containing the metadata
    path_img
        path to the directory containing the augmented images
    path_data
        path to the directory containing the augmented metadata
    verbose
        verbose flag to display images while generating them
    """
    # define dataset reader
    reader = JSONReader(args.root_dir, metadata, frame_rate=frame_rate)

    # load pose estimation if necessary
    pose_estimator = None
    if args.use_pose:
        pose_estimator = PoseEstimation(360, 640, 'pose/ckpts/exp_pose_model_best.pth.tar')

    # extract scene name and reset frame index
    scene = metadata.split(".")[0]
    frame_idx = 0

    # read the first frame from the video
    frame, speed, rel_course = reader.get_next_image()
    while True:
        try:
            # get next frame corresponding to current prediction
            next_frame, next_speed, next_rel_course = reader.get_next_image()
        except Exception as e:
            break

        if next_frame.size == 0:
            break
        
        if (rel_course is None) or (abs(speed) < 1e-3):
            frame = next_frame
            speed = next_speed
            rel_course = next_rel_course
            continue

        # save image
        frame_path = os.path.join(path_img, scene + "." + str(frame_idx).zfill(5) + ".png")
        cv2.imwrite(frame_path, frame[..., ::-1])

        # save data
        data_path = os.path.join(path_data, scene + "." + str(frame_idx).zfill(5) + ".pkl")
        with open(data_path, "wb") as fout:
            if args.use_pose:
                # use pose estimator for labeling
                pose = pose_estimator.get_pose(frame, next_frame, (0, 0, 0.1, 0.1))
                
                # this radius is not scaled
                R = pose_estimator.compute_radius_from_pose(pose)
            else:
                # compute turning radius from relative course
                R = steering.get_radius_from_course(rel_course, speed, 1.0 / frame_rate)

            # save data
            pkl.dump({"speed": speed, "rel_course": rel_course, "radius": R, "frame_rate": frame_rate}, fout)

        # update frame count
        frame_idx += 1

        if verbose:
            print("Speed: %.2f, 1/R: %.2f" % (speed, 1./R))
            print("Frame shape:", frame.shape)
            cv2.imshow("Frame", frame[..., ::-1])
            cv2.waitKey(0)

        # update frame, speed, relative course
        frame = next_frame
        speed = next_speed
        rel_course = next_rel_course


if __name__ == "__main__":
    args = parse_args()

    # create parent directory
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    # create subdirectory for the old/new dataset
    path = os.path.join("dataset", "pose_dataset" if args.use_pose else "gt_dataset")
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
    files = os.listdir(args.root_dir)
    metadata = [file for file in files if file.endswith(".json")]
    
    # process all scenes
    for md in tqdm(metadata):
        read_data(
            metadata=md,
            path_img=path_img,
            path_data=path_data,
            frame_rate=args.frame_rate,
            verbose=args.verbose
        )

