import numpy as np
import logging

CAR_L = 2.634  # Wheel base
CAR_T = 1.497  # Tread

MIN_TURNING_RADIUS = 5.
MAX_STEER = 500
MAX_WHEEL_ANGLE = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
STEERING_RATIO = MAX_STEER / MAX_WHEEL_ANGLE
eps = 1e-8


def sign(x: float) -> int:
    """"
    Returns the sign of x. 1 for numbers greather or
    equal than 0, -1 for numbers less than 0

    Parameters
    ----------
    x
        number for which we want to get the sign

    Returns
    -------
    Sign of the given number
    """
    return 1 if x >= 0 else -1


def get_delta_from_steer(steer: float, steering_ratio: float = STEERING_RATIO) -> float:
    """
    Computes the angle of the imaginary wheel (see Ackerman model)
    given the current angle on the steering wheel

    Parameters
    ----------
    steer
        angle of the steering wheel in degrees
    steering_ration
        ratio of maximum steer and maximum wheel angle (is constant)

    Returns
    -------
    Angle of the imaginary wheel (see Ackerman model)
    """

    sgn = sign(steer)
    delta = sgn * min(MAX_WHEEL_ANGLE, abs(steer) / steering_ratio)
    return delta


def get_steer_from_delta(delta: float, steering_ratio: float = STEERING_RATIO) -> float:
    """
    Computes the angle on the steering wheel
    given the angle of the imaginary wheel (see Ackerman model)

    Parameters
    ----------
    delta
        angle on the imaginary wheel
    steering_ratio
        ratio of maximum steer and maximum wheel angle (is constant)

    Returns
    -------
    Angle on the steering wheel
    """
    sgn = sign(delta)
    steer = sgn * min(MAX_STEER, abs(delta) * steering_ratio)
    return steer


def get_radius_from_delta(delta: float, car_l: float = CAR_L) -> float:
    """
    Computes the trajectory radius from the angle
    of the imaginary wheel

    Parameters
    ----------
    delta
        angle on the imaginary wheel
    car_l
        wheel base

    Returns
    -------
    radius of the circle that the car makes
    """
    sgn = sign(delta)
    r = car_l / np.tan(np.deg2rad(abs(delta), dtype=np.float32) + eps)
    r = sgn * max(r, MIN_TURNING_RADIUS)
    return r


def get_delta_from_radius(r, car_l=CAR_L, car_t=CAR_T):
    """
    Computs the angle of the imaginary wheel given
    the radius of the trajectory

    Parameters
    ----------
    r
        turning radius ( calculated against back center)
    car_l
        wheel base
    car_t
        tread

    Returns
    -------
    Angles of front center, inner wheel, outer wheel
    """
    sgn = sign(r)
    r = max(abs(r), MIN_TURNING_RADIUS)
    delta_i = sgn * np.rad2deg(np.arctan(car_l / (r - car_t / 2.)))
    delta = sgn * np.rad2deg(np.arctan(car_l / r))
    delta_o = sgn * np.rad2deg(np.arctan(car_l / (r + car_t / 2.)))
    return delta, delta_i, delta_o


def get_steer_from_course(course: float, speed: float, dt: float = 0.33, eps: float = 1e-8):
    """
    Computs the angle of the steering wheel given
    the relative course, speed and time(dt). After
    dt, car has a relative course given as argument.
    We want to compute the steering that will get
    us into the same position after dt.

    Parameters
    ----------
    course
        relative course
    speed
        speed of the car in m/s
    dt
        delta time. default dt=0.33
        (that's how the network was trained)
    eps
        numerical stability

    Returns
    -------
    The angle on the steering wheel that will get us
    into the same relative orientation after dt seconds
    """
    if abs(speed) < 1e-6:
        logging.warning("Can not convert steer from course when" \
                        "the speed is 0. Just return 0.")
        return 0.0

    sgn = np.sign(course)
    dist = speed * dt
    R = dist / (np.deg2rad(abs(course)) + eps)
    delta, _, _ = get_delta_from_radius(R)
    steer = sgn * get_steer_from_delta(delta)
    return steer


def get_course_from_steer(steer: float, speed: float, dt: float = 0.33):
    """
    Computes the relative course given the angel of the
    steering angle, speed and dt. For more details
    see the description for the inverse "get_steer_from_course"

    Parameters
    ----------
    steer
        angle fo the steering wheel
    speed
        speed of the car in m/s
    dt
        delta time. default dt=0.33
        (that's how the network was trained)

    Returns
    -------
    The relative course after dt seconds
    """
    if abs(speed) < 1e-6:
        logging.warning("Can not convert course from steer when" \
                        "the speed is 0. Just return 0.")
        return 0

    dist = speed * dt
    delta = get_delta_from_steer(steer)
    R = get_radius_from_delta(delta)
    R = max(R, MIN_TURNING_RADIUS)
    rad_course = dist / R
    course = np.rad2deg(rad_course)
    return course


def get_radius_from_course(course: float, speed: float, dt: float = 0.33):
    """
    Computes turning radius from relative course

    Parameters
    ----------
    course
        relative course
    speed
        speed of the vehicle in m/s
    dt
        delta time; default=0.33s

    Returns
    -------
    Turning radius
    """
    steer = get_steer_from_course(course, speed, dt)
    delta = get_delta_from_steer(steer)
    R = get_radius_from_delta(delta)
    return R
