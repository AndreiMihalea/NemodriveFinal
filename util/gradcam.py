import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
import cv2
from torch.autograd import Variable, backward
from copy import deepcopy


class GradCAM(object):
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer

        # set forward hook function
        # and define storing variable
        self.output_layer = None
        self.layer.register_forward_hook(self._hook_forward)
        
        # set backward hook function
        self.grad_layer = None
        self.layer.register_backward_hook(self._hook_backward)


    def _hook_forward(self, module, input, output):
        self.output_layer = output[0].clone()

    def _hook_backward(self, module, grad_in, grad_out):
        self.grad_layer = grad_out[0].clone()

    def __call__(self, data: dict, idx: int)->np.array:
        # zero grad and set model to eval
        self.model.zero_grad()
        self.model.eval()

        # transform input image to variable
        # that requires gradient
        self.data = deepcopy(data)
        self.data["img"] = Variable(data["img"], requires_grad=True)
        

        # send it to model's device
        self.data["img"] = self.data["img"].to(self.model.device)
        self.data["speed"] = self.data["speed"].to(self.model.device)

        # pass model through the model
        output =  self.model(self.data)

        # define mask
        mask = torch.zeros_like(output)
        mask[0, idx] = 1

        # compute gradients
        output.backward(gradient=mask)

        # average gradients to get the weights
        # for each feature map
        weights = torch.mean(self.grad_layer, dim=(2, 3))
        weights = weights.reshape(*weights.shape, 1, 1)

        # compute the final activation map
        amap = weights * self.output_layer
        amap = amap.sum(dim=1)
        
        # reshape the activation map
        # to match the input image shape
        h, w = data["img"].shape[2:]
        amap = amap.unsqueeze(0)
        amap = F.interpolate(amap, mode='bilinear', size=(h, w), align_corners=True)

        # transform activation map to numpy array
        amap = amap.squeeze(0).detach().cpu().numpy()
        amap = np.transpose(amap, (1, 2, 0))

        # normalize actiavtion map
        amax, amin = amap.max(), amap.min()
        amap = (amap - amin) / (amax - amin)
        return amap


    @staticmethod
    def cam_vis(img: np.array, amap: np.array, weight: float = 0.3) -> np.array:
        """
        :param img: input image, values in range [0, 1]
        :param amap: activation map, values in range [0, 1]
        :param weight: weighting factor between the image an activation map
        :return: overlapping between input and activation, values in range [0, 255], uint8
        """
        # get colormap
        amap = (amap * 255).astype(np.uint8)
        amap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
        amap = amap[...,::-1]

        # combine colormap and image
        img  = (255 * img).copy()
        combined = weight * img + (1 - weight) * amap
        return combined.astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image
    import os
    import models
    import util
    import cv2

    for i in range(100):
        # read input image and transfrom to numpy
        img = Image.open("dataset/old_dataset/img_real/d9837004c66f44f4.000" + str(i).zfill(2) + ".png")
        img = np.asarray(img)
    
        # process image
        img = img / 255.
        timg = img.transpose((2, 0, 1))
        timg = torch.tensor(timg).unsqueeze(0)
        timg = timg.float()

        # define model
        nbins = 401
        model = models.RESNET(
            no_outputs=nbins,
            use_speed=True,
            use_old=True
        )
        model.to(model.device)

        # load ckpt model
        path = os.path.join("snapshots", "00000", "ckpts", "default.pth")
        util.load_ckpt(path, [('model', model)])
        model.eval()
    
        # pass image through the model
        tspeed = torch.tensor([[2.]]).float()
        data = {
                "img": timg.to(model.device), 
                "speed": tspeed.to(model.device)
        }
        output = model(data)
    
        # get the most probable action
        idx = torch.argmax(output).item()

        # define GradCAM object
        gcam = GradCAM(model, model.encoder[-1][-1])
        amap = gcam(data, idx)

        # get visualization
        vis = GradCAM.cam_vis(img, amap)
        img = (255 * img).astype(np.uint8)
        full = np.concatenate([img, vis], axis=1)
        cv2.imshow("IMG", full[..., ::-1])
        cv2.waitKey(0)

