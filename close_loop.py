from models.resnet import *
from util.io import *

import argparse
import itertools
import random
import os
import pickle as pkl
from tqdm import tqdm

import torch.nn
import torch.nn.functional as F

from util.reader import JSONReader
from simulator.evaluator import AugmentationEvaluator
from simulator.plots import *

import util.vis as vis
from util.vis import gaussian_dist, normalize, unnormalize
from typing import Tuple


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin', type=int, help="starting video index", default=0)
    parser.add_argument('--end', type=int, help="ending video index", default=81)
    parser.add_argument('--load_model', type=str, help='name of the model', default='default')
    parser.add_argument('--use_speed', action='store_true', help='use speed of the vehicle')
    parser.add_argument('--use_old', action='store_true', help="use old dataset")
    parser.add_argument('--split_path', type=str, help='path to the file containing the test scenes (test_scenes.txt)')
    parser.add_argument('--data_path', type=str, help='path to the directory of raw dataset (.mov & .json)')
    parser.add_argument('--sim_dir', type=str, help='simulation directory', default='simulation')
    args = parser.parse_args()
    return args


def set_seed(seed: int = 0):
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def process_frame(frame):
    frame = normalize(frame)

    # transpose and change shape
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame).unsqueeze(0).float().cuda()
    return frame



def moving_average(x, w):
    """
    Signal smoothing

    Parameters
    ----------
    x
        signal to be smoothed
    w
        window

    Returns
    -------
    Smoothed signal
    """
    return np.convolve(x, np.ones(w), 'same') / w


def get_turning(output, smooth=True) -> Tuple[float, np.ndarray]:
    """
    Computes the turning radius (actually  1/R) from the
    output distribution

    Parameters
    ----------
    output
        predicted distribution
    smooth
        boolean flag to apply smoothing filter

    Returns
    -------
    Most probable curvature and the smoothed output
    """
    if smooth:
        output = moving_average(output, 5)
        output /= output.sum()

    index = np.argmax(output).item()
    return (index - 200) / 1000, output


def make_prediction(frame: np.ndarray, speed: float) -> Tuple[float, torch.tensor]:
    """
    Make prediction given the current observation

    Parameters
    ----------
    frame
        RGB observation
    speed
        current speed in [m/s]

    Returns
    -------
    Most probable turning radius (actually 1/R) and the tensor distribution
    """

    # preprocess data
    frame = process_frame(frame)
    speed = torch.tensor([[speed]]).to(device)

    # make first prediction
    with torch.no_grad():
        # construct data package
        data = {
            "img": frame.to(device),
            "speed": speed.to(device),
        }

        # make prediction based on frame
        toutput = model(data)

        # process the logits and get the course as the argmax
        toutput = F.softmax(toutput, dim=1)
        output = toutput.reshape(toutput.shape[1]).cpu().numpy()
        course, output = get_turning(output)
        toutput = torch.tensor(output.reshape(*toutput.shape)).to(device)

    return course, toutput


def test_video(root_dir: str, metadata: str, time_penalty=6,
               translation_threshold=1.5, rotation_threshold=0.2, verbose=True):

    """ Close loop evaluation of a video """

    scene = metadata.split('.')[0]
    log_path = os.path.join(args.sim_dir, args.load_model, scene)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, "imgs"))

    # buffers to store evaluation details
    turnings = []
    predicted_turnings = []

    predicted_turning_distributions = []
    turning_distributions = []

    # initialize reader
    reader = JSONReader(root_dir=root_dir, json_file=metadata, frame_rate=3)

    # initialize evaluators
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    augm = AugmentationEvaluator(
        reader=reader,
        time_penalty=time_penalty,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold
    )

    # get first two frames of the video and make a prediction
    frame, speed, turning, interv = augm.reset()

    with torch.no_grad():
        for idx in itertools.count():
            # video is done
            if frame.size == 0:
                break
            
            # make prediction
            pred_turning, toutput = make_prediction(frame, speed)
            next_frame, next_speed, next_turning, interv = augm.step(pred_turning)
            
            if not interv:
                # distribution for the ground truth course
                turning_distribution = gaussian_dist(1000 * turning + 200)
                turning_distributions.append(turning_distribution)

                # distribution for the predicted course
                predicted_turning_distribution = gaussian_dist(1000 * pred_turning + 200)
                predicted_turning_distributions.append(predicted_turning_distribution)

                # save values
                turnings.append(turning)
                predicted_turnings.append(pred_turning)

                # construct full image
                turning_distribution = torch.tensor(turning_distribution).unsqueeze(0)
                imgs_path = os.path.join(args.sim_dir, args.load_model, scene, "imgs", str(idx).zfill(5) + ".png")
                full_img = visualisation(process_frame(frame), turning_distribution, toutput,
                                         1, imgs_path).astype(np.uint8)
            else:
                # signla intervention by a blank screen
                full_img = np.zeros((vis.HEIGHT, 3 * vis.WIDTH, 3)).astype(np.uint8)

            # save image
            pil_snapshot = pil.fromarray(full_img.astype(np.uint8))
            pil_snapshot.save(imgs_path)

            if verbose:
                    print("Pred_Turning: %.3f, Turning: %.3f, Speed: %.2f" % (pred_turning, turning, speed))
                    cv2.imshow("State", full_img[..., ::-1])
                    cv2.waitKey(100)

            # update frame
            frame = next_frame
            turning = next_turning
            speed = next_speed

    # get some statistics [mean distance till an intervention, mean angle till an intervention]
    statistics = augm.statistics
    absolute_mean_distance, absolute_mean_angle, plot_dist_ang = plot_statistics(statistics)

    # save stats plot
    stats_path = os.path.join(args.sim_dir, args.load_model, scene, "stats.png")
    cv2.imwrite(stats_path, plot_dist_ang[..., ::-1])

    # save all data
    data = {
        "turning": turnings,
        "predicted_courses": predicted_turnings,
        "predicted_course_distributions": predicted_turning_distributions,
        "real_course_distributions": turning_distributions,
        "autonomy": augm.autonomy,
        "num_interventions": augm.number_interventions,
        "video_length": augm.video_length,
        "statistics": statistics,
    }

    data_path = os.path.join(args.sim_dir, args.load_model, scene, "data.pkl")
    with open(data_path, 'wb') as fout:
        pkl.dump(data, fout)


if __name__ == "__main__":
    # get arguments
    args = get_args()

    # set seed
    set_seed(0)

    # get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    nbins = 401
    experiment = ''
    model = RESNET(no_outputs=nbins).to(device)
    experiment += "resnet"
    model = model.to(device)

    # load model
    path = os.path.join("snapshots", args.load_model, "ckpts", "default.pth")
    load_ckpt(path, [('model', model)])
    model.eval()

    # construct simulation dirs
    if not os.path.exists(args.sim_dir):
        os.mkdir(args.sim_dir)

    # get list of test scenes
    with open(args.split_path, 'rt') as fin:
        files = fin.read()

    files = files.strip().split("\n")
    files = [file + ".json" for file in files]
    files = files[args.begin:args.end]

    # test video
    for file in tqdm(files[:5]):
        test_video(root_dir=args.data_path, metadata=file, verbose=False)
