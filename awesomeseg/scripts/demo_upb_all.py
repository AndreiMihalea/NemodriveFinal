from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from PIL import Image

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

    def eval(self):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        driving_dirs = os.listdir(args.demo_dir)

        for current_dir in driving_dirs:
            if '.txt' in current_dir:
                continue
            pics = []
            current_dir_path = os.path.join(args.demo_dir, current_dir)
            files = os.listdir(current_dir_path)
            pics = sorted([os.path.join(current_dir_path, x) for x in files if '.png' in x])[:-1]

            for i, input_pic in enumerate(pics):
                print(input_pic)
                image = Image.open(input_pic)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = crop_to_size(image, img_height, img_width)
                # image = cv2.warpAffine(image, np.float32([[1, 0, -90], [0, 1, -110]]), (image.shape[1], image.shape[0]))
                image = self.input_transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = model(image)

                logits = (outputs[0][0][0]).cpu().data.numpy()
                print(outputs[0][0][0].shape)
            
                logits[logits > 0.35] = 1
                logits[logits != 1] = 0

                output = torch.tensor(logits).to(self.device)

                mask_overlay = np.array(logits)
                mask_overlay = np.array([mask_overlay] * 3)
                mask_overlay[mask_overlay != 0] = 255
                mask_overlay[mask_overlay != 255] = 0
                mask_overlay[0, :, :] = 0
                mask_overlay[2, :, :] = 0

                img = cv2.imread(input_pic)
                # img = cv2.resize(img, (832, 256))[:, 96:-96, :]
                # print(img.shape, mask_overlay.shape)
                res_img = cv2.addWeighted(mask_overlay.transpose((1, 2, 0)).astype(np.float32), 1., img.astype(np.float32), 1, 0)
                # cv2.imshow('res', res_img / 255.0)
                # cv2.waitKey(1)

                if self.args.save_pred:
                    pred = torch.argmax(outputs[0], 1)
                    pred = pred.cpu().data.numpy()

                    predict = pred.squeeze(0)
                    save_helper = input_pic.split('/')[-2:]
                    seq_dir, frame = save_helper[0], save_helper[1]
                    print(seq_dir, frame)
                    # print(os.path.join(outdir, '\\'.join(filename[0].split('/')[-3:])))
                    # np.save(os.path.join(outdir, '\\'.join(filename[0].split('/')[-3:])), logits)
                    # mask = get_color_pallete(predict, self.args.dataset)
                    seq_save_dir = os.path.join(outdir, seq_dir)
                    if not os.path.exists(seq_save_dir):
                        os.makedirs(seq_save_dir)
                    torch.save(outputs[0][0][0], os.path.join(seq_save_dir, frame).replace('.png', '.pt'))
                    # mask.save(os.path.join(seq_save_dir, frame))
            synchronize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation Demo With Pytorch')
    # model and dataset
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
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # demo dir
    parser.add_argument('--demo-dir', type=str, default='/mnt/storage/workspace/andreim/nemodrive/upb_data/dataset/test_frames/',
                        help='demo directory')  

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_tensor_all/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            print(outdir)
            os.makedirs(outdir)


    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
