from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torchvision.models as models


class RESNET(nn.Module):
    def __init__(self, no_outputs, use_speed=False, use_old=True, use_roi=None):
        super(RESNET, self).__init__()
        self.no_outputs = no_outputs
        self.use_speed = use_speed
        self.use_old   = use_old
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_roi = use_roi

        if self.use_roi == 'input':
            self.input_channels = 4
        else:
            self.input_channels = 3
        
        # construct encoder
        rnet = models.resnet18(pretrained=True)
        conv1 = nn.Conv2d(self.input_channels, rnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(
            conv1,
            rnet.bn1,
            rnet.maxpool,
            rnet.layer1,
            rnet.layer2,
            rnet.layer3,
            rnet.layer4,
        )

        # construct encoder for rgb input
        rnet_input = models.resnet18(pretrained=True)
        conv1_input = nn.Conv2d(self.input_channels, rnet_input.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.encoder_input = nn.Sequential(
            conv1_input,
            rnet_input.bn1,
        )

        # construct encoder for roi
        self.encoder_roi = nn.Conv2d(1, rnet_input.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.encoder_roi.bias.data = torch.ones(self.encoder_roi.bias.data.shape)
        self.encoder_roi.weight.data = torch.zeros(self.encoder_roi.weight.data.shape)

        # construct rest of network for rgb + roi features
        self.encoder_input_roi = nn.Sequential(
            rnet_input.maxpool,
            rnet_input.layer1,
            rnet_input.layer2,
            rnet_input.layer3,
            rnet_input.layer4,
        )
        
        # adaptive avg pool
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # classifier
        self.fc1 = nn.Linear(512 + (1 if self.use_speed else 0), 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512 + (1 if self.use_speed else 0), self.no_outputs)


    def forward(self, data):
        B, _, H, W = data["img"].shape

        # mean and standard deviation for rgb image
        if self.use_old:
            mean_rgb = torch.tensor([0.49, 0.45, 0.47]).view(1, 3, 1, 1).to(self.device)
            std_rgb  = torch.tensor([0.15, 0.15, 0.16]).view(1, 3, 1, 1).to(self.device)
        else:
            mean_rgb = torch.tensor([0.57, 0.44, 0.30]).view(1, 3, 1, 1).to(self.device)
            std_rgb  = torch.tensor([0.31, 0.30, 0.25]).view(1, 3, 1, 1).to(self.device)

        # make input unit normal
        img = data["img"]
        img = (img - mean_rgb) / std_rgb

        roi = data["roi"]

        # feature extractor
        if self.use_roi == 'features':
            input = self.encoder_input(img)
            roi = self.encoder_roi(roi)
            input = input * roi
            input = self.encoder_input_roi(input)
        else:
            input = self.encoder(img)
        
        # average pooling
        input = self.avgpool(input)
        input = input.reshape(input.shape[0], -1)

        # append speed if necessary
        if self.use_speed:
            input = torch.cat([input, data["speed"]], dim=1)
        
        # pass through the classifier
        input = self.fc1(input)
        input = self.relu1(input)
        input = self.dp1(input)

        if self.use_speed:
            input = torch.cat([input, data["speed"]], dim=1)

        # output probability distribution
        output = self.fc2(input)
        return output
