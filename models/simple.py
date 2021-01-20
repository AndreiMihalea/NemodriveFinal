import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple(nn.Module):
    def __init__(self, no_outputs):
        super(Simple, self).__init__()
        
        # define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # number of outputs
        self.no_outputs = no_outputs

        # define architecture
        self.conv1 = nn.Conv2d(3,  24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(17920, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, no_outputs)


    def forward(self, data):
        B, _, H, W = data["img"].shape

        # mean and standard deviation for rgb image
        mean_rgb = torch.tensor([0.49, 0.45, 0.47]).view(1, 3, 1, 1).to(self.device)
        std_rgb  = torch.tensor([0.15, 0.15, 0.16]).view(1, 3, 1, 1).to(self.device)
        
        # make input unit normal
        img = data["img"]
        img = (img - mean_rgb) / std_rgb

        x = F.relu(self.conv1(img), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.flatten(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x

