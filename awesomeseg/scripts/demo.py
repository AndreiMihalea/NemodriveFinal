import numpy as np

def demo(config):
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
    # model.load_state_dict(torch.load('/home/andrei/.torch/models/fcn32s_vgg16_kitti.pth'))
    # print(model)
    print('Finished loading model!')

    with open(args.val_file, 'r') as f:
        pics = f.readlines()
    pics = [x.split(',')[0] for x in pics]

    print(len(pics))

    # This is only for saving stuff # COMMENT WHEN NOT USING
    pics = []
    files = os.listdir('/home/andrei/Documents/research_semester_4/segmentation_upb_hard')
    pics = [x.replace('\\', '/').replace('.png', '.jpg') for x in files]

    model.eval()
    with torch.no_grad():
        for i, input_pic in enumerate(pics):
            if i % 1 == 0:
                print(input_pic)
                image = Image.open(input_pic)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = crop_to_size(image, img_height, img_width)
                # image = cv2.warpAffine(image, np.float32([[1, 0, -90], [0, 1, -110]]), (image.shape[1], image.shape[0]))
                images = transform(image).unsqueeze(0).to(device)
                output = model(images)

                # print(output[0][0][0][:5][:5], output[0][0][1][:5][:5])

                for act in [(torch.nn.Sigmoid(), 'sigmoid', 0.28), (torch.nn.Softmax(), 'softmax', 0.015), (None, 'logits', 0.15)]:
                    if act[0] is None:
                        pred = output[0].squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        pred = act[0](output[0].squeeze(0).squeeze(0)).cpu().numpy()
                    outname = os.path.splitext(os.path.split(input_pic)[-1])[0] + '.png'
                    # outname = outname.split('_')[0] + '_lane_' + outname.split('_')[1] + '.png'
                    # mask.save(os.path.join(args.outdir, outname))

                    mask = np.zeros((pred.shape[0], pred.shape[1], 3)).astype(np.uint8)
                    mask[pred[:, :] > act[2]] = 255
                    mask[:, :, 0] = 0
                    mask[:, :, 2] = 0
                    # outname = os.path.splitext(os.path.split(input_pic)[-1])[0] + '.png'
                    # outname = outname.split('_')[0] + '_lane_' + outname.split('_')[1]
                    # cv2.imwrite(os.path.join(args.outdir, outname), mask)

                    image_res = cv2.addWeighted(mask, 1, image, 1, 0)
                    # cv2.imshow("img", image_res)
                    # cv2.waitKey(0)

                    cv2.imwrite('/home/andrei/Documents/research_semester_4/segmentation_upb_soft/' +
                                input_pic.replace('/', '\\').replace('.jpg', '_{}.png'.format(act[1])), image_res)
                    # torch.save(pred, '/home/andrei/Documents/research_semester_4/upb_soft_eval_fine_tuned/' +
                    #            input_pic.replace('/', '\\').replace('.jpg', '.pt'))

                    # heatmap
                    logits = output[0][0][0].cpu().data.numpy()
                    logits = logits

                    plt.figure(figsize=(6.4, 2.88), dpi=100)
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    sns.heatmap(pred, cbar=False, xticklabels=False, yticklabels=False)
                    outname_heatmap = '/home/andrei/Documents/research_semester_4/segmentation_upb_soft/' + \
                                input_pic.replace('/', '\\').replace('.jpg', '_heatmap_{}.png'.format(act[1]))
                    plt.savefig(outname_heatmap)
                    plt.close()


if __name__ == '__main__':
    demo(args)import os
import sys
import argparse
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc/pascal_aug/ade20k/citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the predict result')
args = parser.parse_args()


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
    print('Finished loading model!')

    model.eval()
    with torch.no_grad():
        output = model(images)

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))


if __name__ == '__main__':
    demo(args)
