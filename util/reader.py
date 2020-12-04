import os
import cv2
import utm
import json
import nemodata
import numpy as np
import matplotlib.pylab as plt

from .vis import *
from .transformation import Crop
from .transformation import Convertor
from .steering import *
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

    @staticmethod
    def get_course(steer, speed, dt):
        dist = speed * dt
        delta = get_delta_from_steer(steer)
        R = get_radius_from_delta(delta)
        rad_course = dist / R
        course = np.rad2deg(rad_course)
        return course



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
        # self.prev_packet = self.get_package()

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
        steer = packet["sensor_data"]["canbus"]["steer"]["value"]
        return center_img, speed, steer

    def get_next_image(self):
        packet = self.get_package()
        
        if len(packet[0]) == 0:
            return packet

        # TODO compute relative course
        # rel_course = JSONReader.get_relative_course(self.prev_packet[2], new_packet[2])
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


if __name__ == "__main__":
    use_old_data = False
    old_dir = "/home/robert/PycharmProjects/upb_dataset"
    old_file = "0e49f41acc2b428e.json"
    

    dirs = [
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/forward/13_28_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/forward/17_32_11_03_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/forward/17_37_11_03_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/forward/18_05_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/reverse/13_34_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/reverse/13_38_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/reverse/17_44_11_03_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/reverse/17_50_11_03_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/reverse/18_12_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/sport_biotehnice_rectorat_automatica/forward/18_21_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/sport_biotehnice_rectorat_automatica/reverse/18_37_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica_biotehnice_rectorat/forward/17_56_11_03_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_energetica/forward/17_56_10_12_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice_energetica/forward/19_00_10_29_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice_energetica/forward/17_40_10_12_2020',
 #'/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice_energetica/reverse/19_07_10_29_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice_energetica/reverse/17_47_10_12_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice/forward/18_17_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice/forward/17_00_10_12_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice/forward/18_00_10_12_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice/reverse/18_11_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_biotehnice/reverse/17_16_10_12_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/forward/14_09_10_29_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/forward/13_55_10_29_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/forward/18_25_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/forward/18_31_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/forward/18_43_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/reverse/14_18_10_29_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/reverse/14_03_10_29_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/reverse/18_37_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/rectorat_sport/reverse/18_48_11_03_2020',
 '/media/nemodrive/Samsung_T5/nemodrive_upb2020/energetica_automatica/forward/18_53_10_29_2020']

    for new_dir in dirs:

        #new_dir =  '/media/nemodrive/Samsung_T5/nemodrive_upb2020/automatica/forward/13_28_10_29_2020'
        new_file = "metadata.pkl"
        reader = JSONReader(old_dir, old_file) if use_old_data else PKLReader(new_dir, new_file)

        while True:
            # get next frame corresponding to current prediction
            # frame, _, _ = reader.get_next_image()
            x = reader.get_next_image()

            if x[0].size == 0:
                break

        #print("speed:", x[1], "rel_course:", x[2])
        # cv2.imshow("Center image", x[0])
        # cv2.waitKey(0)
        # print("==========")
