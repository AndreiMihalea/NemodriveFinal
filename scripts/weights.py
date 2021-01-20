#!/usr/bin/env python
# coding: utf-8

import argparse
from tqdm import tqdm
from util.dataset import *

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--nclasses", type=int, default=5, help="number of bins for data balancing")
parser.add_argument("--use_pose", action="store_true", help="use pose estimation dataset")
args = parser.parse_args()

# define dataset
path_dataset = os.path.join("dataset", "pose_dataset" if args.use_pose else "gt_dataset")
train_dataset = UPBDataset(path_dataset, train=True)


def get_index(x, a_min: float, a_max: float, nclasses: int):
	assert a_min >= 0 and a_max > 0, "bounds should be positive"
	assert a_min < a_max, "invalid bounds"

	eps = 1e-6
	x = np.abs(x)
	x = np.clip(x, a_min, a_max - eps)
	grid = np.linspace(a_min, a_max, nclasses + 1)
	return np.sum(x >= grid) - 1


def compute_weights(nclasses=5):
	""" function to compute weights """
	global train_dataset
	
	counts = [0] * nclasses
	turning = [0] * len(train_dataset)
	
	for i in tqdm(range(len(train_dataset))):
		turning[i] = get_index(train_dataset[i]['turning'].item(), 0, 0.2, nclasses)
		counts[turning[i]] += 1
		
	weights_per_class = [0.] * nclasses
	N = np.sum(counts)
	for i in range(nclasses):
		if counts[i] > 0:
			weights_per_class[i] = N / float(nclasses * counts[i])

	weights = [0] * len(train_dataset)
	for i in tqdm(range(len(turning))):
		weights[i] = weights_per_class[turning[i]]
	return weights


if __name__ == "__main__":
	# compute weights
	weights = compute_weights(nclasses=args.nclasses)

	# data sampler
	tf_weights = torch.DoubleTensor(weights)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(tf_weights, len(tf_weights))
	train_loader = DataLoader(
		train_dataset,
		batch_size=256,
		sampler=sampler,
		num_workers=4,
		pin_memory=True
	)

	# display a batch
	data = next(iter(train_loader))
	turning = data['turning'].numpy().reshape(-1)
	turning = np.abs(turning)
	sns.distplot(turning, bins=args.nclasses)
	plt.show()

	# save to csv
	path_weights = os.path.join(path_dataset, "weights")
	path_weights += ".csv"

	df = pd.DataFrame(data=weights, columns=["name"])
	df.to_csv(path_weights, index=False)

	# double check by reading the file
	df_weights = pd.read_csv(path_weights)
	df_weights.head()







