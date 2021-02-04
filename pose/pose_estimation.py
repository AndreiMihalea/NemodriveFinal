# Contains a class for loading the pose estimation model
import os
import torch
import pose.models
import numpy as np
import pandas as pd
from imageio import imread
from skimage.transform import resize as imresize


class PoseEstimation(object):
    def __init__(self, img_height, img_width, path):
        """
        Constructor

        Parameters
        ----------
        img_height
            desired image height
        img_width
            desired image width
        path
            path to the pretrained pose estimator model
        """
        self.pose_net = None
        self.camera_matrix = None
        self.img_height = img_height
        self.img_width = img_width

        # define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the pretrained pose esimtaro
        self.load_model(path)

    def load_model(self, path):
        self.pose_net = pose.models.PoseNet().to(self.device)
        weights_pose = torch.load(path)
        self.pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
        self.pose_net.eval()

    def load_pose(self, path):
        self.pose_df = pd.read_csv(path, sep=" ", header=None)
        return self.pose_df

    def preprocess_img(self, img, crop_left=0, crop_right=0, crop_bot=0, crop_top=0):
        """
        Parameters
        ----------
        img
            original image
        crop_left
            percentage to crop on the left side of the image
        crop_right
            percentage to crop on the right side of the image
        crop_bot
            percentage to crop on the bottom side of the image
        crop_top
            percentage to crop on the top side of the image
        Returns
        -------
        Processed image
        """
        # resize the image to the desired shape
        proc_img = imresize(img, (self.img_height, self.img_width)).astype(np.float32)

        # get image height and width
        img_height, img_width, _ = proc_img.shape

        # compute the new sizes
        new_top = int(crop_top * img_height)
        new_bot = img_height - int(crop_bot * img_height)
        new_left = int(crop_left * img_width)
        new_right = img_width - int(crop_right * img_width)

        # crop the image
        proc_img = proc_img[new_top:new_bot, new_left:new_right]

        # transform it to tensor
        proc_img = np.transpose(proc_img, (2, 0, 1))
        proc_img = ((torch.from_numpy(proc_img).unsqueeze(0) - 0.5) / 0.5).to(self.device)
        return proc_img

    def get_pose(self, frame: np.ndarray, next_frame: np.ndarray, crops: tuple = (0, 0, 0, 0)) -> torch.tensor:
        """
        Parameters
        ----------
        frame
            frame at time t
        next_frame
            frame at time t + 1
        crops
            tuple containing the percentage to be cropped
            from the image as: crop_left, crop_right, crop_bot, crop_top
        Returns
        -------
        Pose prediction containing the translation and rotation on each axis
        """
        crop_left, crop_right, crop_bot, crop_top = crops
        tensor_img1 = self.preprocess_img(frame, crop_left=crop_left, crop_right=crop_right,
                                          crop_bot=crop_bot, crop_top=crop_top)
        tensor_img2 = self.preprocess_img(next_frame, crop_left=crop_left, crop_right=crop_right,
                                          crop_bot=crop_bot, crop_top=crop_top)

        with torch.no_grad():
            pose = self.pose_net(tensor_img1, tensor_img2)

        return pose

    def compute_radius_from_pose(self, pose: torch.tensor):
        """"
        Computes turning radius form pose estimation

        Parameters
        ----------
        pose
            torch.tensor containing the scale translation and the euler angles

        Returns
        -------
        Turning radius
        """
        pose = pose.view(-1).cpu().numpy()
        assert len(pose) == 6
        
        # extract translation on x and z axis
        # and the rotation on y axis
        tx, _,  tz = pose[:3]
        _, ry, _ = pose[3:]

        # compute distance from the origin to
        # the next point
        d = np.sqrt(tx**2 + tz**2)

        # compute turning radius using
        # generealized Pitagora's theorem
        eps = 1e-12
        R = -np.sign(ry) * d / np.sqrt(2 * max(1 - np.cos(ry), eps))
        return R
