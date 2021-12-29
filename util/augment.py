import cv2
import numpy as np
import math

from simulator import steering
from simulator import transformation

from .reader import Reader
from typing import Tuple


class PerspectiveAugmentator(object):
    @staticmethod
    def compute_command(data, translation, rotation, intersection_distance=10):
        """
        Computes the augmented steering command
        Warning!!! this augmentation may work only for turns less than 180 degrees. For bigger turns, although it
        reaches the same point, it may not follow the real car's trajectory

        Parameters
        ----------
        data
            [steer, velocity, delta_time]
        translation
            ox translation, be aware that positive values mean right translation
        rotation
            rotation angle, be aware that positive valuea mean right rotation
        intersection_distance
            distance where the simulated car and real car will intersect

        Returns
        -------
        The augmented frame, steer for augmented frame
        """
        assert abs(rotation) < math.pi / 2, "The angle in absolute value must be less than Pi/2"

        steer, _, _ = data
        eps = 1e-12

        # compute wheel angle and radius of the real car
        steer = eps if abs(steer) < eps else steer
        wheel_angle = steering.get_delta_from_steer(steer + eps)
        R = steering.get_radius_from_delta(wheel_angle)

        # estimate the future position of the real car
        alpha = intersection_distance / R  # or may try velocity * delta_time / R
        P1 = np.array([R * (1 - np.cos(alpha)), R * np.sin(alpha)])

        # determine the point where the simulated car is
        P2 = np.array([translation, 0.0])

        # compute the line parameters that passes through simulated point and is
        # perpendicular to it's orientation
        d = np.zeros((3,))
        rotation = eps if abs(rotation) < eps else rotation
        d[0] = np.sin(rotation)
        d[1] = np.cos(rotation)
        d[2] = -d[0] * translation

        # we need to find the circle center (Cx, Cy) for the simulated car
        # we have the equations
        # (P11 - Cx)**2 + (P12 - Cy)**2 = (P21 - Cx)**2 + (P22 - Cy)**2
        # d0 * Cx + d1 * Cy + d2 = 0
        # to solve this, we substitute Cy with -d0/d1 * Cx - d2/d1
        a = P1[0] ** 2 + (P1[1] + d[2] / d[1]) ** 2 - P2[0] ** 2 - (P2[1] + d[2] / d[1]) ** 2
        b = -2 * P2[0] + 2 * d[0] / d[1] * (P2[1] + d[2] / d[1]) + 2 * P1[0] - 2 * d[0] / d[1] * (P1[1] + d[2] / d[1])
        Cx = a / b
        Cy = -d[0] / d[1] * Cx - d[2] / d[1]
        C = np.array([Cx, Cy])

        # determine the radius
        sim_R = np.linalg.norm(C - P2)
        assert np.isclose(sim_R, np.linalg.norm(C - P1)), "The points P1 and P2 are not on the same circle"

        # determine the "sign" of the radius
        # sgn = 1 if np.cross(w2, w1) >= 0 else -1
        w1 = np.array([np.sin(rotation), np.cos(rotation)])
        w2 = P1 - P2
        sgn = 1 if np.cross(w2, w1) >= 0 else -1
        sim_R = sgn * sim_R

        # determine wheel angle
        sim_delta, _, _ = steering.get_delta_from_radius(sim_R)
        sim_steer = steering.get_steer_from_delta(sim_delta)
        return sim_steer, sim_delta, sim_R, C

    @staticmethod
    def pipeline(reader: Reader, img: np.array, tx: float = 0.0, ry: float = 0.0):
        # convention
        ry = -ry

        # transform image to tensor
        img = np.asarray(img)
        height, width = img.shape[:2]

        K = reader.K.copy()
        K[0, :] *= width
        K[1, :] *= height

        M = reader.M.copy()
        M = np.linalg.inv(M)[:3, :]

        # transformation object
        # after all transformation, for the old dataset we end up
        transform = transformation.Transformation(K, M)
        output = transform.rotate_image(img, ry)
        output = transform.translate_image(output, tx)
        return output

    @staticmethod
    def augment(reader: Reader, frame: np.ndarray, roi_map: np.ndarray, R: float,
            speed: float,  frame_rate: int=3, 
            transf: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Augments a given image with the corresponding command
        by sampling at random the translation and the rotation

        Parameters
        ----------
        reader
            JSON reader for the UPB dataset
        frame
            image to be augmented
        roi_map
            region of interest to be augmented
        R
            turning radius
        speed
            vehicle's speed [m/s]
        frame_rate
            video's frame rate
        transf
            tuple containing the transformation applied,
            tx - translation on ox axis, and 
            ry - rotation on oy axis
        
        Returns
        -------
        Tuple containing the augmented image, and turning radius
        """
        # make conversion form relative course to steering
        dt = 1.0 / frame_rate
        delta, _, _ = steering.get_delta_from_radius(r=R)
        steer = steering.get_steer_from_delta(delta)

        # sample the translation and rotation applied
        tx, ry = 0.0, 0.0
        sgnt = 1 if np.random.rand() > 0.5 else -1
        sgnr = 1 if np.random.rand() > 0.5 else -1
        
        # unpack the transformation if sent as a parameter
        # else generate a random one
        if transf is not None:
            tx, ry = transf
        else:
            # generate random transformation
            if np.random.rand() < 0.33:
                tx = sgnt * np.random.uniform(0.5, 1.2)
                ry = sgnr * np.random.uniform(0.05, 0.12)
            else:
                if np.random.rand() < 0.5:
                    tx = sgnt * np.random.uniform(0.5, 1.5)
                else:
                    ry = sgnr * np.random.uniform(0.05, 0.25)

        # generate augmented image
        aug_img = PerspectiveAugmentator.pipeline(
            reader=reader, img=frame, tx=tx, ry=ry)

        if roi_map is not None:
            aug_roi_map = PerspectiveAugmentator.pipeline(
                reader=reader, img=roi_map, tx=tx, ry=ry)
        else:
            aug_roi_map = None

        # cv2.imshow('roi', np.concatenate((roi_map, aug_roi_map)))
        # cv2.waitKey(0)

        # cv2.imshow('frame', np.concatenate((frame, aug_img)))
        # cv2.waitKey(0)
        
        # generate augmented steering command
        aug_steer, _, aug_R, _ = PerspectiveAugmentator.compute_command(
            data=[steer, speed, dt],
            translation=tx,
            rotation=ry,
        )
        
        # compute augmentation course
        aug_course = steering.get_course_from_steer(aug_steer, speed, dt)

        return aug_img, aug_roi_map, aug_R, aug_course
