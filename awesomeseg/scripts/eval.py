from __future__ import print_function

import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

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
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='test', mode='val', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

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

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        for threshold in np.arange(0.1, 1.5, 0.025):          
            self.metric.reset()
            self.model.eval()
            if self.args.distributed:
                model = self.model.module
            else:
                model = self.model
            # logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

            num = 0
            sum_acc = 0.
            sum_miou = 0.
            for i, (image, target, path_masks, filename) in enumerate(self.val_loader):
                if i < 1100:
                    continue

                #print(filename)

                image = image.to(self.device)
                target = target.to(self.device)

                with torch.no_grad():
                    outputs = model(image)
                
                logits = nn.Sigmoid()(outputs[0][0][0])
                logits = logits.cpu().data.numpy()
                softmax = logits.copy()

                # print(target.shape, logits.shape, image.shape, filename)
            
                logits[logits > threshold] = 1
                logits[logits != 1] = 0

                output = torch.tensor(logits).to(self.device)

                pixAcc = torch.sum(torch.eq(output, 1 - target)).item() / target.nelement()
                mIoU = torch.logical_and(output, 1 - target).sum() / torch.logical_or(output, 1 - target).sum()

                num += 1
                sum_acc += pixAcc
                sum_miou += mIoU

                #logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                #       i + 1, pixAcc * 100, mIoU * 100))


                #plt.figure(figsize=(6.4, 2.88), dpi=100)
                #plt.gca().set_axis_off()
                #plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                #               hspace=0, wspace=0)
                #plt.margins(0, 0)
                #plt.gca().xaxis.set_major_locator(plt.NullLocator())
                #plt.gca().yaxis.set_major_locator(plt.NullLocator())
                #sns.heatmap(softmax, cbar=False)#, cbar=True, xticklabels=True, yticklabels=True)
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close()

                #save_path = os.path.join(outdir, '\\'.join(filename[0].split('/')[-3:]))
                #plt.savefig(save_path.replace('pred_pic', 'pred_sigmoid'))
                #plt.close()

                if self.args.save_pred:
                    mask_overlay = np.array(logits)
                    mask_overlay = np.array([mask_overlay] * 3)
                    mask_overlay[mask_overlay != 0] = 255
                    mask_overlay[mask_overlay != 255] = 0
                    mask_overlay[0, :, :] = 0
                    mask_overlay[2, :, :] = 0

                    img = cv2.imread(filename[0])
                    img = cv2.resize(img, (832, 256))[:, 96:-96, :]
                    #print(img.shape, mask_overlay.shape)
                    #res_img = cv2.addWeighted(mask_overlay.transpose((1, 2, 0)).astype(np.float32), 1., img.astype(np.float32), 1, 0)
                    #cv2.imshow('res', res_img / 255.0)
                    #cv2.waitKey(0)
                    #cv2.imwrite(os.path.join(outdir, '\\'.join(filename[0].split('/')[-3:])), res_img)
                    # print(outdir, os.path.splitext(filename[0])[0] + '.png')
                    # np.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.npy'), outputs[0][0][0].cpu().data.numpy())
                    #mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
            synchronize()
            print('logits', threshold, sum_acc / num, sum_miou.item() / num)


if __name__ == '__main__':
    args = parse_args()
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
        outdir = '../runs/pred_pic_sigmoid/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
