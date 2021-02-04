import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import itertools
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument("--dst_dir", type=str, default="./results")
parser.add_argument("--sim_dir", type=str, default="./simulation")
parser.add_argument("--time_penalty", type=int, default=6)
args = parser.parse_args()

if __name__ == "__main__":
    # get experiments
    exprs = sorted(os.listdir(args.sim_dir))

    for expr in exprs:
        # get list of videos
        videos_path = os.path.join(args.sim_dir, expr)
        videos = os.listdir(videos_path)

        # create buffers
        num_intervs = []
        auto = []
        videos_length = []
        distances = []
        angles = []
        steps2interv = []

        for video in tqdm(videos):
            # read data for each video
            path = os.path.join(videos_path, video, "data.pkl")
            with open(path, 'rb') as fin:
                data = pkl.load(fin)

            # compute autonomy for the current video
            ni = data['num_interventions']
            vl = data['video_length']
            a = 1 - (ni * args.time_penalty) / (vl + ni * args.time_penalty)
            auto.append(a)

            num_intervs.append(ni)
            videos_length.append(vl)
            distances += list(itertools.chain.from_iterable(data['statistics']['distances']))
            angles += list(itertools.chain.from_iterable(data['statistics']['angles']))
            steps2interv += [len(x) for x in data['statistics']['distances']]

        # defin path
        expr_path = os.path.join(args.dst_dir, expr)
        if not os.path.exists(expr_path):
            os.makedirs(expr_path)

        # save individual results
        indiv_results = {
            "video": videos,
            "autonomy": auto,
            "interventions": num_intervs,
            "video_length": videos_length,
        }
        df = pd.DataFrame(indiv_results)
        df = df.sort_values("autonomy")
        df.to_csv(os.path.join(expr_path, "detailed_data.csv"))

        # compute global results
        total_videos_length = np.sum(videos_length)
        total_interv = np.sum(num_intervs)

        intervention_time = total_interv * args.time_penalty
        autonomy =  1 - intervention_time / (total_videos_length + intervention_time)
        mean_distance = np.abs(distances).mean()
        std_distance = np.abs(distances).std()
        mean_angle = np.abs(angles).mean()
        std_angle = np.abs(angles).std()
        mean_steps2interv = np.mean(steps2interv)
        
        
        results = {
            "autonomy": autonomy,
            "interventions": total_interv,
            "total videos length": total_videos_length,
            "intevention_time": intervention_time,
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "mean_angle": mean_angle,
            "std_angle": std_angle,
            "steps2interv": mean_steps2interv
        }
        df = pd.DataFrame([results])
        df.to_csv(os.path.join(expr_path, "data.csv"))

        print("Experiment: %s" % (expr))
        print("Autonomy = %.2f" % (autonomy))
        print("Interventions = %d" % (total_interv))
        print("Mean Distance = %.2f, Std Distance = %.2f" % (mean_distance, std_distance))
        print("Mean Angle = %.2f, Std Angle = %.2f" % (mean_angle, std_angle))
        print("Mean Steps2Interv = %.2f" % (mean_steps2interv))
        print("MD / MS2I = %.5f" % (mean_distance / mean_steps2interv))
        print("MA / MS2I = %.5f" % (mean_angle / mean_steps2interv))
        print("Videos Time = %.2f" % (total_videos_length))
        print("Videos Time + Time Penalty = %.2f" % (total_videos_length + intervention_time))
        print("====" * 10)        
    
