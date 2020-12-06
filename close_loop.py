from models.resnet import *
from util.io import *
from util.vis import *

import torch
import torch.nn.functional as F

import os
import argparse
import itertools
import pickle as pkl
from tqdm import tqdm
import numpy as np
import random

from util.reader import Reader, JSONReader, PKLReader
from util.evaluator import AugmentationEvaluator
from util.plots import *


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

# set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# get available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
nbins = 401
experiment = ''
model = RESNET(
    no_outputs=nbins,
    use_speed=args.use_speed,
    use_old=args.use_old    
).to(device)
experiment += "resnet"
model = model.to(device)

# load model
path = os.path.join("snapshots", args.load_model, "ckpts", "default.pth")
load_ckpt(path, [('model', model)])
model.eval()

# construct simulation dirs
if not os.path.exists(args.sim_dir):
    os.mkdir(args.sim_dir)


# construct gaussian distribution
def gaussian_distribution(mean=200.0, std=5, eps=1e-6):
    x = np.arange(401)
    mean = np.clip(mean, 0, 400)

    # construct pdf
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2)
    pmf = pdf / (pdf.sum() + eps)
    return pmf


# processing frame
def normalize(img):
    return img / 255.


def unnormalize(img):
    return (img * 255).astype(np.uint8)


def process_frame(frame):
    frame = normalize(frame)

    # transpose and change shape
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame).unsqueeze(0).float().cuda()
    return frame


# output smoothing
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def get_course(output, smooth=True):
    if smooth:
        output = moving_average(output, 5)
        output /= output.sum()

    index = np.argmax(output).item()
    return (index - 200) / 10, output


def make_prediction(frame, speed):
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
        course, output = get_course(output)
        toutput = torch.tensor(output.reshape(*toutput.shape)).to(device)

    return course, toutput


# close loop evaluation
def test_video(root_dir: str, metadata: str, time_penalty=6,
               translation_threshold=1.5, rotation_threshold=0.2, verbose=True):

    scene = metadata.split('.')[0]
    log_path = os.path.join(args.sim_dir, args.load_model, scene)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, "imgs"))

    # buffers to store evaluation details
    real_courses = []
    predicted_courses = []
    predicted_course_distributions = []
    real_course_distributions = []

    # initialize reader
    if args.use_old:
        reader = JSONReader(root_dir=root_dir, json_file=metadata, frame_rate=3)
    else:
        path = os.path.join(root_dir, metadata)
        reader = PKLReader(root_dir=path, pkl_file="metadata.pkl", frame_rate=3)

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
    frame, speed, real_course = augm.reset()

    with torch.no_grad():
        for idx in itertools.count():
            # video is done
            if frame.size == 0:
                break
            
            frame = frame[..., ::-1] # BGR to RGB
            pred_course, toutput = make_prediction(frame, speed)
            next_frame, next_speed, next_real_course = augm.step(pred_course)
            
            if real_course is not None:
                # distribution for the ground truth course
                real_course_distribution = gaussian_distribution(10 * (real_course + 20))
                real_course_distributions.append(real_course_distribution)

                # distribution for the predicted course
                predicted_course_distribution = gaussian_distribution(10 * (pred_course + 20))
                predicted_course_distributions.append(predicted_course_distribution)

                output = toutput.reshape(toutput.shape[1]).cpu().numpy()
                predicted_courses.append(output)
                real_courses.append(real_course)

                # construct full image
                real_course_distribution = torch.tensor(real_course_distribution).unsqueeze(0)
                imgs_path = os.path.join(args.sim_dir, args.load_model, scene, "imgs", str(idx).zfill(5) + ".png")
                full_img = visualisation(process_frame(frame), real_course_distribution, toutput, 1,
                                         imgs_path).astype(np.uint8)

                # print and save courses
                if verbose:
                    print("Predicted Course: %.2f, Real Course: %.2f, Speed: %.2f" % (pred_course, real_course, speed))
                    cv2.imshow("State", full_img[..., ::-1])
                    cv2.waitKey(100)

            # update frame
            frame = next_frame
            real_course = next_real_course
            speed = next_speed

    # get some statistics [mean distance till an intervention, mean angle till an intervention]
    statistics = augm.statistics
    absolute_mean_distance, absolute_mean_angle, plot_dist_ang = plot_statistics(statistics)

    # save stats plot
    stats_path = os.path.join(args.sim_dir, args.load_model, scene, "stats.png")
    cv2.imwrite(stats_path, plot_dist_ang[..., ::-1])

    # save all data
    data = {
        "real_courses": real_courses,
        "predicted_courses": predicted_courses,
        "predicted_course_distributions": predicted_course_distributions,
        "real_course_distributions": real_course_distributions,
        "autonomy": augm.autonomy,
        "num_interventions": augm.number_interventions,
        "video_length": augm.video_length,
        "statistics": statistics,
    }

    data_path = os.path.join(args.sim_dir, args.load_model, scene, "data.pkl")
    with open(data_path, 'wb') as fout:
        pkl.dump(data, fout)


if __name__ == "__main__":
    with open(args.split_path, 'rt') as fin:
        files = fin.read()

    files = files.strip().split("\n")
    if args.use_old:
        files = [file + ".json" for file in files]   
    files = files[args.begin:args.end]

    # test video
    for file in tqdm(files):
        test_video(root_dir=args.data_path, metadata=file, verbose=False)
