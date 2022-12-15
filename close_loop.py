from models.resnet import *
from models.pilot import *

from util.io import *

import argparse
import itertools
import random
import os
import pickle as pkl
from tqdm import tqdm

import torch.nn
import torch.nn.functional as F
from torchvision import transforms

from util.reader import JSONReader
from simulator.evaluator import AugmentationEvaluator
from simulator.plots import *

import util.vis as vis
from util.vis import gaussian_dist, normalize, unnormalize, normalize_with_neg
from typing import Tuple
from scipy.signal import find_peaks

from util.car_utils import get_radius, WHEEL_STEER_RATIO, render_path, render_path_lines
from util.segmentation_utils import compute_score, compute_miou

import sys

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

sys.path.insert(1, '/home/nemodrive/workspace/andreim/awesome-semantic-segmentation-pytorch')

from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

segmentation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


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
    parser.add_argument('--use_baseline', action='store_true', help='predict 0 degrees always')
    # segmentation arguments
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k',
                                 'citys', 'sbu', 'upb', 'kitti'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument("--use_roi", choices=['none', 'input', 'features'], help="use path region of interest as input")
    parser.add_argument("--roi_map", choices=['seg_hard', 'seg_soft', 'gt_hard', 'gt_soft'], default='seg_soft',
                        help="use path region of interest as input")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def get_segmentation_network():
    BatchNorm2d = nn.BatchNorm2d
    model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                   aux=args.aux, pretrained=True, pretrained_base=False,
                                   local_rank=args.local_rank,
                                   norm_layer=BatchNorm2d).to(device)
    return model


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


def make_prediction(frame: np.ndarray, roi_map: np.ndarray, speed: float) -> Tuple[float, torch.tensor]:
    """
    Make prediction given the current observation

    Parameters
    ----------
    frame
        RGB observation
    roi_map
        region of interest to feed to the network
    speed
        current speed in [m/s]

    Returns
    -------
    Most probable turning radius (actually 1/R) and the tensor distribution
    """

    # preprocess data
    frame = process_frame(frame)
    speed = torch.tensor([[speed]]).to(device)
    roi_map = torch.tensor(roi_map).unsqueeze(0).unsqueeze(0)

    # make first prediction
    with torch.no_grad():
        # construct data package
        data = {
            "img": frame.to(device),
            "speed": speed.to(device),
            "roi": roi_map.to(device)
        }

        # make prediction based on frame
        toutput = model(data)

        # process the logits and get the course as the argmax
        if args.use_baseline:
            toutput = torch.tensor(gaussian_dist(200, std=10)).unsqueeze(0).float()
        else:
            toutput = F.softmax(toutput, dim=1)
        output = toutput.reshape(toutput.shape[1]).cpu().numpy()
        course, output = get_turning(output)
        toutput = torch.tensor(output.reshape(*toutput.shape)).to(device)

    return course, toutput


def test_video(root_dir: str, metadata: str, time_penalty=6,
               translation_threshold=1.5, rotation_threshold=0.2, verbose=True):

    """ Close loop evaluation of a video """

    segmentation_model = get_segmentation_network()
    segmentation_model.eval()

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
    reader_seg = JSONReader(root_dir=root_dir, json_file=metadata, frame_rate=3)

    # initialize evaluators
    # check multiple parameters like time_penalty, distance threshold and angle threshold
    # in the original paper time_penalty was 6s
    augm = AugmentationEvaluator(
        reader=reader,
        time_penalty=time_penalty,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold
    )

    augm_seg = AugmentationEvaluator(
        reader=reader_seg,
        time_penalty=time_penalty,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold,
        process_input = False
    )

    # get first two frames of the video and make a prediction
    frame, speed, turning, interv = augm.reset()
    frame_seg, _, _, _ = augm_seg.reset()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))

    num = 501
    MAX_STEER = 500
    angles = np.linspace(-MAX_STEER, MAX_STEER, num)

    with torch.no_grad():
        for idx in itertools.count():
            # video is done
            if frame.size == 0:
                break

            # segmentation of the current frame
            current_frame_seg = frame_seg.copy()
            current_frame_seg = cv2.cvtColor(current_frame_seg, cv2.COLOR_BGR2RGB)
            current_frame_seg = segmentation_transform(current_frame_seg).unsqueeze(0).to(device)

            with torch.no_grad():
                # print(segmentation_model(current_frame_seg)[0].shape)
                output = segmentation_model(current_frame_seg)
                roi_map = nn.Sigmoid()(output[0][0][0]).cpu().numpy()
                for _ in range(1):
                    roi_map = cv2.morphologyEx(roi_map, cv2.MORPH_CLOSE, kernel)
                # roi_map[roi_map > 0.001] = 1
                # roi_map[roi_map != 1] = 0
                # roi_map = cv2.morphologyEx(roi_map, cv2.MORPH_OPEN, kernel)
                # plt.imshow(roi_map)
                # plt.show()
                # aug_roi_map_weight = np.stack([roi_map] * 3).transpose((1, 2, 0)).astype(float)
                # aug_roi_map_weight[:, :, 0] = 0
                # aug_roi_map_weight[:, :, 2] = 0
                # frame_roi = cv2.addWeighted(frame_seg.astype(float), 1, aug_roi_map_weight * 255., 0.5, 0)
                # cv2.imshow('img', frame_roi/255.)
                # cv2.waitKey(0)


            roi_map = augm_seg.force_process_frame(roi_map)
            roi_map = normalize_with_neg(roi_map)

            if 'hard' in args.roi_map:
                pred = 1 - torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                pred = augm_seg.force_process_frame(pred)
                # print(pred.shape)
                # plt.imshow(pred)
                # plt.colorbar()
                # plt.show()
                # roi_map = pred


            # make prediction
            pred_turning, toutput = make_prediction(frame, roi_map, speed)

            noutput = toutput.cpu().numpy()[0]
            peaks, _ = find_peaks(noutput, height=0.005, distance=10)

            max_score = 0
            best_peak = 0

            image_ackermann = frame_seg.copy()

            image = frame_seg.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = segmentation_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = segmentation_model(image)

            logits = nn.Sigmoid()(outputs[0][0][0]).cpu().data.numpy()
            for _ in range(1):
                logits = cv2.morphologyEx(logits, cv2.MORPH_CLOSE, kernel)
            logits_og = logits.copy()

            logits[logits > 0.35] = 1
            logits[logits != 1] = 0

            mask_overlay = np.array(logits)
            mask_overlay = np.array([mask_overlay] * 3)
            mask_overlay[mask_overlay != 0] = 255
            mask_overlay[mask_overlay != 255] = 0


            # img = cv2.resize(img, (832, 256))[:, 96:-96, :]
            # print(img.shape, mask_overlay.shape)
            # res_img = cv2.addWeighted(mask_overlay.transpose((1, 2, 0)).astype(np.float32), 1., frame_segm.astype(np.float32), 1, 0)
            # cv2.imshow('res', res_img / 255.0)
            # cv2.waitKey(0)

            for peak in angles:#peaks: # peaks if only want to consider the distribution peaks (angles if want to consider all possibilities)
                turning_seg = (peak - 200) / 1000
                pred_steer = -augm.get_pred_steer(turning_seg)
                # print(pred_steer, 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                radius = get_radius(pred_steer / WHEEL_STEER_RATIO)

                image_res, overlay = render_path(frame_seg, radius, augm.camera_position)
                overlay_ackermann = overlay[:, :, 1] / overlay.max()
                image_ackermann = render_path_lines(image_ackermann, radius, augm.camera_position)

                miou = compute_miou(overlay_ackermann, mask_overlay)
                soft_score = compute_score(overlay_ackermann, logits_og)

                score = soft_score

                if score > max_score:
                    max_score = score
                    best_peak = turning_seg

            if len(peaks) > 1 and best_peak != pred_turning and abs(pred_turning - turning) > abs(best_peak - turning):
                print(best_peak * 1000 + 200)
                plt.figure(figsize=(6.4, 3.6), dpi=100)
                plt.xlabel("Bin index")
                plt.ylabel("Bin value")
                # plt.gca().set_axis_off()
                plt.subplots_adjust(left=0.135, bottom=0.135, right=0.99, top=0.99, wspace=0.1, hspace=0.1)
                plt.margins(0.03, 0.03)
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.plot([x for x in range(len(noutput))], noutput)
                plt.plot(peaks, noutput[peaks], "x", mew=3, ms=10)
                plt.show()
                cv2.imshow("final", image_ackermann)
                cv2.waitKey(0)
                turning_seg = (best_peak - 200) / 1000
                pred_steer = -augm.get_pred_steer(turning_seg)
                radius = get_radius(pred_steer / WHEEL_STEER_RATIO)
                image_res, overlay = render_path(frame_seg, radius, augm.camera_position)
                heatmapshow = cv2.normalize(logits_og, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_HOT)
                # cv2.imshow("final", np.concatenate((image_ackermann, heatmapshow)))
                cv2.imshow("final", np.concatenate((cv2.cvtColor(image_ackermann, cv2.COLOR_BGR2RGB), heatmapshow)))
                # cv2.imwrite('paper_files/seg_paths.png', np.concatenate((cv2.cvtColor(image_ackermann, cv2.COLOR_BGR2RGB), heatmapshow)))
                # cv2.imwrite('paper_files/seg_paths_horizontal.png', np.concatenate((cv2.cvtColor(image_ackermann, cv2.COLOR_BGR2RGB), heatmapshow), axis=1))
                # cv2.imwrite('paper_files/seg_paths_solo.png', cv2.cvtColor(image_ackermann, cv2.COLOR_BGR2RGB))
                # cv2.imwrite('paper_files/seg_paths_heatmap.png', heatmapshow)
                cv2.waitKey(0)
                plt.figure(figsize=(6.4, 3.6), dpi=100)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                   hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                sns.heatmap(logits_og, cbar=False)
                plt.show()

            next_frame, next_speed, next_turning, interv = augm.step(best_peak)
            next_frame_seg, _, _, _ = augm_seg.step(best_peak)
            
            if not interv:
                # distribution for the ground truth course
                turning_distribution = gaussian_dist(1000 * turning + 200, std=10)
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
                # full_img = visualisation(process_frame(frame), turning_distribution, toutput,
                #                          1, imgs_path).astype(np.uint8)
            else:
                # signla intervention by a blank screen
                pass
                full_img = np.zeros((vis.HEIGHT, 2 * vis.WIDTH, 3)).astype(np.uint8)

            # save image
            # pil_snapshot = pil.fromarray(full_img.astype(np.uint8))
            # pil_snapshot.save(imgs_path)

            if verbose:
                print("Pred_Turning: %.3f, Turning: %.3f, Speed: %.2f" % (pred_turning, turning, speed))
                print("%.2f, %.2f" % (augm.simulator.distance, augm.simulator.angle))
                # cv2.imshow("State", full_img[..., ::-1])
                # cv2.waitKey(0)

            # update frame
            frame = next_frame
            frame_seg = next_frame_seg
            turning = next_turning
            speed = next_speed

    # get some statistics [mean distance till an intervention, mean angle till an intervention]
    statistics = augm.statistics
    #absolute_mean_distance, absolute_mean_angle, plot_dist_ang = plot_statistics(statistics)

    # save stats plot
    stats_path = os.path.join(args.sim_dir, args.load_model, scene, "stats.png")
    #cv2.imwrite(stats_path, plot_dist_ang[..., ::-1])

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
        "intervention_coords": augm.get_intervention_points()
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
    model = RESNET(no_outputs=nbins, use_roi=args.use_roi).to(device)
    # model = PilotNet(no_outputs=nbins).to(device)
    model = model.to(device)

    # load model
    if not args.use_baseline:
        path = os.path.join("snapshots_pose", args.load_model, "ckpts", "default.pth")
        print(path)
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
    files = files[args.begin:args.end]#[24:]

    # test video
    for file in tqdm(files):
        test_video(root_dir=args.data_path, metadata=file, verbose=False)
