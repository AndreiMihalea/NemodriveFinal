import os
import cv2
import utm
import json
import nemodata
import numpy as np
import matplotlib.pylab as plt

from util.vis import *
from util.transformation import Crop
from nemodata.compression import Decompressor


class Reader(object):
    def __init__(self, root_dir: str, file: str, frame_rate: int):
        self.root_dir = root_dir
        self.file = file
        self.frame_rate = frame_rate

        # initial location
        self.init_northing = None
        self.init_easting = None

        # location at the current time
        self.northing = None
        self.easting = None

    def get_next_image(self):
        raise NotImplemented()

    @property
    def init_location(self):
        return self.init_northing, self.init_easting

    @property
    def location(self):
       return self.northing, self.easting

    @property
    def K(self):
        raise NotImplemented()

    @property
    def M(self):
        raise NotImplemented()

    @staticmethod
    def crop_car(img: np.array):
        raise NotImplemented()

    @staticmethod
    def crop_center(img: np.array):
        raise NotImplemented()

    @staticmethod
    def resize_img(img: np.array):
        raise NotImplemented()

    @property
    def video_length(self):
        raise NotImplementedError()


class JSONReader(Reader):
    def __init__(self, root_dir: str, json_file: str, frame_rate: int=3):
        """
        :param root_dir: root directory of the dataset
        :param json_file: file name
        :param frame_rate: frame rate of the desired dataset
        """
        super(JSONReader, self).__init__(root_dir, json_file, frame_rate)
        self._read_json()
        self.reset()

    def _read_json(self):
        # get data from json
        with open(os.path.join(self.root_dir, self.file)) as f:
            self.data = json.load(f)

        # get cameras
        self.center_camera = self.data['cameras'][0]

        # read locations list
        self.locations = self.data['locations']

    def reset(self):
        video_path = os.path.join(self.root_dir, self.file[:-5] + ".mov")
        self.center_capture = cv2.VideoCapture(video_path)
        self.frame_index = -1
        self.locations_index = 0


    @property
    def K(self):
        """ Returns intrinsic camera matrix """
        return np.array([
            [0.61, 0, 0.5],  # width
            [0, 1.09, 0.5],  # height
            [0, 0, 1]])

    @property
    def M(self):
        """ Returns extrinsic camera matrix """
        return np.array([
            [1,  0, 0, 0.00],
            [0, -1, 0, 1.65],
            [0,  0, 1, 1.54],
            [0,  0, 0, 1]
        ])

    @staticmethod
    def crop_car(img: np.array):
        return img[:320, ...]

    @staticmethod
    def crop_center(img: np.array):
        return Crop.crop_center(img, up=0.1, down=0.5, left=0.25, right=0.25)

    @staticmethod
    def resize_img(img: np.array):
        return cv2.resize(img, dsize=None, fx=0.8, fy=0.8)

    @staticmethod
    def get_relative_course(prev_course, crt_course):
        a = crt_course - prev_course
        a = (a + 180) % 360 - 180
        return a

    @property
    def video_length(self):
        return (self.frame_index + 1) / self.frame_rate

    def _get_closest_location(self, tp):
        return min(self.locations, key=lambda x: abs(x['timestamp'] - tp))

    def get_next_image(self):
        """
        :param predicted_course: predicted course by nn in degrees
        :return: augmented image corresponding to predicted course or empty np.array in case the video ended
        """
        ret, frame = self.center_capture.read()
        self.frame_index += 1
        dt = 1. / self.frame_rate

        # check if the video ended
        if not ret:
            return np.array([]), None, None

        # read course and speed for previous frame
        location = self._get_closest_location(1000 * dt * self.frame_index + self.locations[0]['timestamp'])
        next_location = self._get_closest_location(1000 * dt * (self.frame_index + 1) + self.locations[0]['timestamp'])
 
        # compute relative course and save current course
        rel_course = JSONReader.get_relative_course(location['course'], next_location['course'])
        speed = location['speed']

        # # initialize locations
        # # TODO
        # if not self.init_northing:
        #     self.init_northing = location['northing']
        #     self.init_easting = location['easting']
        #
        # # update location
        # self.northing = location['northing']
        # self.easting = location['easting']

        return frame, speed, rel_course


def gaussian_dist(mean=200.0, std=5, nbins=401):
    x = np.arange(401)
    pdf = np.exp(-0.5 * ((x - mean) / std)**2)
    pmf = pdf / pdf.sum()
    return pmf


class PKLReader(Reader):
    def __init__(self, root_dir: str, pkl_file: str = None, frame_rate: int = 3):
        super(PKLReader, self).__init__(root_dir, pkl_file, frame_rate)
        self.root_dir = root_dir
        self.pkl_file = pkl_file
        self.frame_rate = frame_rate
        self.frame_idx = -1
        self.generator = self._create_generator()
        self.prev_packet = self.get_package()

    def _create_generator(self):
        dt = int(1.0 / self.frame_rate * 1000)
        player = nemodata.VariableSampleRatePlayer(self.root_dir, min_packet_delay_ms=dt)
        player.start()
        return player.stream_generator()

    @property
    def K(self):
        """ Returns intrinsic camera matrix """
        return np.array([
            [0.95, 0,  0.5],  # width
            [0, 1.27, 0.55],  # height
            [0,  0, 1]])

    @property
    def M(self):
        """ Returns extrinsic camera matrix """
        return np.array([
            [1, 0, 0, 0.00],
            [0, -1, 0, 1.65],
            [0, 0, 1, 1.54],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def crop_car(img: np.array):
        return img[:380, ...]

    @staticmethod
    def crop_center(img: np.array):
        return Crop.crop_center(img, up=0.0, left=0.25, right=0.25)

    @staticmethod
    def resize_img(img: np.array):
        return cv2.resize(img, dsize=None, fx=0.8, fy=0.8)

    @property
    def video_length(self):
        return (self.frame_idx + 1) / self.frame_rate

    def get_package(self):
        try:
            packet = next(self.generator)
            self.frame_idx += 1
        except Exception as e:
            return np.array([]), None, None

        # get image
        center_img = packet["images"]["center"]
        center_img = cv2.resize(center_img, None, fx=0.3, fy=0.3)

        # read GPS coordinates
        # TODO
        # if 'gps' in packet['sensor_data'] and 'RMC' in packet['sensor_data']['gps']:
        #     lat_N = float(packet['sensor_data']['gps']['RMC'].latitude)
        #     lon_E = float(packet['sensor_data']['gps']['RMC'].longitude)
        #
        #     # transform GPS coordinates to UTM (northing, easting)
        #     easting, northing, _, _ = utm.from_latlon(latitude=lat_N, longitude=lon_E)
        #
        #     # initialize the location
        #     if not self.init_northing:
        #         self.init_northing = northing
        #         self.init_easting = easting
        #
        #     # update northing & easting
        #     self.northing = northing
        #     self.easting = easting
        #
        #     print(f'Northing: {self.northing}, Easting: {self.easting}')

        # # get rotation
        # vals = packet["sensor_data"]["imu"]["orientation_quaternion"]
        # q = [vals['x'], vals['y'], vals['z'], vals['w']]
        # e = quat.quat2deg(q)

        # get speed value km/h
        speed = packet["sensor_data"]["canbus"]["speed"]["value"]
        return center_img, speed, 0

    def get_next_image(self):
        new_packet = self.get_package()
        if len(new_packet[0]) == 0:
            return new_packet

        # TODO compute relative course
        # rel_course = JSONReader.get_relative_course(self.prev_packet[2], new_packet[2])
        rel_course = 0
        img = self.prev_packet[0]
        speed = self.prev_packet[1]

        # plotter
        dist = gaussian_dist(200 + 10 * rel_course)
        fig = plt.figure()
        plt.plot(dist)
        course_img = np.asarray(fig2img(fig, img.shape[1], img.shape[0]))
        plt.close(fig)
        full_img = np.concatenate([img, course_img], axis=1)
        print("speed", speed, "rel_course", rel_course)
        cv2.imshow("FULL", full_img)
        cv2.waitKey(0)

        self.prev_packet = new_packet
        return img, speed, rel_course


if __name__ == "__main__":
    use_old_data = False
    old_dir = "/home/robert/PycharmProjects/upb_dataset"
    old_file = "0e49f41acc2b428e.json"

    new_dir = "/home/robert/PycharmProjects/upb_dataset_new/video1/"
    new_file = "metadata.pkl"
    reader = JSONReader(old_dir, old_file) if use_old_data else PKLReader(new_dir, new_file)

    while True:
        # get next frame corresponding to current prediction
        # frame, _, _ = reader.get_next_image()
        x = reader.get_next_image()

        if x[0].size == 0:
            break

        print("speed:", x[1], "rel_course:", x[2])
        # cv2.imshow("Center image", x[0])
        # cv2.waitKey(0)
        # print("==========")
