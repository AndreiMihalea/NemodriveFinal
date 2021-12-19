import os
import cv2
import json
import numpy as np
from simulator.transformation import Crop

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
    def __init__(self, root_dir: str = None, json_file: str = None, frame_rate: int=3):
        """
        :param root_dir: root directory of the dataset
        :param json_file: file name
        :param frame_rate: frame rate of the desired dataset
        """
        super(JSONReader, self).__init__(root_dir, json_file, frame_rate)

        # define attributes
        self.center_capture = None
        self.frame_index = None
        self.locations_index = None
        self.init_northing = None
        self.init_easting = None

        # initialize objects
        if root_dir:
            self._read_json()
            self.reset()

    def _read_json(self):
        # get data from json
        with open(os.path.join(self.root_dir, self.file)) as f:
            self.data = json.load(f)
        
        # get camera
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
            [0,  1, 0, 1.65],
            [0,  0, 1, 1.54],
            [0,  0, 0, 1]
        ])

    @staticmethod
    def crop_car(img: np.array):
        return img[:320, ...]

    @staticmethod
    def crop_center(img: np.array):
        img = Crop.crop_center(img, up=0.0, down=0.5, left=0.25, right=0.25)
        img = Crop.crop_center(img, up=0.35)
        return img

    @staticmethod
    def resize_img(img: np.array):
        return cv2.resize(img, dsize=(200, 66))

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

        # initialize locations
        if not self.init_northing:
            self.init_northing = location['northing']
            self.init_easting = location['easting']

        # update location
        self.northing = location['northing']
        self.easting = location['easting']

        # update course
        self.course = location["course"]
        
        return frame[..., ::-1], speed, rel_course

class PKLReader(Reader):
    def __init__(self, root_dir: str, pkl_file: str = None, frame_rate: int = 3):
        super(PKLReader, self).__init__(root_dir, pkl_file, frame_rate)
        self.root_dir = root_dir
        self.pkl_file = pkl_file
        self.frame_rate = frame_rate
        self.frame_idx = -1
        self.generator = self._create_generator()

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
            [0, 1, 0, 1.65],
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

        # get speed value km/h
        speed = packet["sensor_data"]["canbus"]["speed"]["value"]
        steer = packet["sensor_data"]["canbus"]["steer"]["value"]
        return center_img, speed, steer


    def get_next_image(self):
        packet = self.get_package()
        if len(packet[0]) == 0:
            return packet

        img, speed, steer = packet

        # remove offset
        OFFSET = -737.6
        steer = None if steer == 0 else steer - OFFSET

        # trasform from km/h to m/s
        speed = Convertor.kmperh2mpers(speed)

        # transform steering to course
        rel_course = None

        if steer:
            dt = 1.0 / self.frame_rate
            rel_course = Reader.get_course(steer, speed, dt)
            rel_course = -rel_course # convention 

        # plotter
        #dist = gaussian_dist(200 + 10 * rel_course)
        #fig = plt.figure()
        #plt.plot(dist)
        #course_img = np.asarray(fig2img(fig, img.shape[1], img.shape[0]))
        #plt.close(fig)
        #full_img = np.concatenate([img, course_img], axis=1)
        #print("speed", speed, "rel_course", rel_course)
        #print("============")
        #cv2.imshow("FULL", full_img)
        #cv2.waitKey(0)

        return img, speed, rel_course
