import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')

sns.set()
from util.vis import *

HEIGHT = 720
WIDTH = 1080


def plot_statistics(statistics):
    """
    plot the statistics received form the simulator
    :param statistics: dict {
        "distances": [[d11, d12, ...], [d21, d22, ...], ...]
        "angles": [[a11, a12, ...], [a21, a22, ...], ...]
    }
    :return: dict {
        "mean_distance": mean over all distances
        "mean_angle": mean over all angles
    }
    """

    all_distances = []
    all_angles = []

    for i in range(len(statistics["distances"])):
        all_distances = all_distances + statistics["distances"][i]

        # transform angles from radians to degrees for better interpretation
        statistics['angles'][i] = list(np.rad2deg(statistics["angles"][i]))
        all_angles = all_angles + statistics["angles"][i]

    # plot distance and angles
    figure = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(all_distances, 'o-')
    plt.title('Distances and angles between the simulated car and the real car')
    plt.ylabel('Distance[m]')
    plt.subplot(2, 1, 2)
    plt.plot(all_angles, '.-')
    plt.xlabel('Frame')
    plt.ylabel('Angle[deg]')
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)

    # compute absolute means
    absolute_mean_distance = 0;
    cnt_mean_distance = 0.
    absolute_mean_angle = 0;
    cnt_mean_angle = 0.

    for i in range(len(statistics["distances"])):
        if len(statistics["distances"][i]) > 0:
            absolute_mean_distance += np.mean(np.abs(statistics["distances"][i]))
            cnt_mean_distance += 1

        if len(statistics["angles"][i]) > 0:
            absolute_mean_angle += np.mean(np.abs(statistics["angles"][i]))
            cnt_mean_angle += 1

    absolute_mean_distance /= cnt_mean_distance
    absolute_mean_angle /= cnt_mean_angle
    return absolute_mean_distance, absolute_mean_angle, plot


def plot_trajectories(trajectories, confidence):
    """
    Plot trajectories of the simulated and real car
    :param trajectories: dict {
        "real_trajectory": [(x1, y1), (x2, y2), ... ]
        "simulated_trajectory:" [(x1, y1), (x2, y2), ..]
    }
    :return: None
    """
    real_trajectory = np.array(trajectories['real_trajectory'])
    simulated_trajectory = np.array(trajectories['sim_trajectory'])

    confidence_points = np.array(confidence['confidence'])
    confidence_points = confidence_points / confidence_points.max()

    figure = plt.figure(figsize=(16, 9), dpi=160)
    plt.scatter(real_trajectory[:, 0], real_trajectory[:, 1],
                c="blue", label="Real car trajectory", s=60)
    plt.scatter(simulated_trajectory[:, 0], simulated_trajectory[:, 1],
                c=confidence_points, label="Simulated car trajectory", s=60, cmap='gray')
    plt.legend(loc=2)
    plt.colorbar()
    plot = np.asarray(fig2img(figure, height=HEIGHT, width=WIDTH))
    plt.close(figure)
    return plot
