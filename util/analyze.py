import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

from util.reader import *
from tqdm import tqdm


if __name__ == "__main__":
    # read files
    PATH = "/media/nemodrive/Samsung_T5/nemodrive_upb2020/"
    files = list(glob.iglob(PATH + "**/*.pkl", recursive=True))
    folders = ["/".join(file.split('/')[:-1]) for file in files]

    # create buffer for steering and direction
    steering_buff = []
    direction_buff = []

    # creat data directory
    if not os.path.exists("steering"):
        os.makedirs("steering")


    for i, folder in tqdm(enumerate(folders)):
        # create data reader
        reader = PKLReader(folder, "metadata.pkl")

        # extract label (forward/revers)
        label = folder.split('/')[-2]

        try:
            while True:
                # get next frame corresponding to current prediction
                # frame, _, _ = reader.get_next_image()
                x = reader.get_next_image()

                if x[0].size == 0:
                    break

                img, speed, rel_course = x

                if rel_course is not None:
                    steering_buff.append(rel_course)
                    direction_buff.append(label)
        except:
            print(f"Exception in folder: {folder}")
    
        # save data
        with open("steering/steering_%d.pkl" % (i, ), "wb") as fout:
            pkl.dump({"directions": direction_buff, "steering": steering_buff }, fout)
    
    # countplot for data in forward/revers
    sns.countplot(direction_buff)
    plt.title("Countplot Direction")
    plt.show()

    # plot steering histogram
    sns.distplot(steering_buff)
    plt.title("Steering Distribution")
    plt.show()

   
