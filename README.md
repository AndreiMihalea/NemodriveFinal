# Steering-Network-1
<p align="center">
  <img src="sample/train_sample.png" alt="train_sample" width="512" />
</p>

## Create dataset

```shell
mkdir raw_dataset
```

* Download the UBP dataset into the "raw_dataset" directory. A sample of the UPB dataset is available <a href="https://drive.google.com/drive/folders/1p_2-_Xo-Wd9MCnkYqPfGyKs2BnbeApqn?usp=sharing">here</a>.

```shell
mkdir scene_splits
```

* Download the scene splits into the "scene_splits" directory. The train-validation split is available <a href="https://github.com/RobertSamoilescu/UPB-Dataset-Split">here</a>.
In the "scene_splits" directory you should have: "train_scenes.txt" and "test_scenes.txt".

```shell
cd scripts

# create the dataset
python3 create_dataset.py --root_dir ../raw_dataset

# split the dataset into train-test
python3 split_dataset.py --train ../scene_splits/train_scenes.txt --test ../scene_splits/test_scenes.txt

# create synthetic dataset by performin 2D perspective augmentations only for the training dataset
python3 create_aug_dataset.py --root_dir ../raw_dataset --train ../scene_splits/train_scenes.txt
```

## Train models
```shell
./run_train.sh
```
  
## Test models - Open-loop evaluation
```shell
./run_test.sh
```

<!--
## Results of the open-loop evaluation
|Model        | Augm.   | Mean  | St. dev.  | Min   | Max   |
|-------------|---------|-------|-----------|-------|-------|
|Baseline     |   *     | 1.797 | 2.504     | 0.035 | 6.874 |
|Simple       |   No    | 0.673 | 0.933     | 0.000 |10.877 | 
|ResNet18     |   No    | 0.645 | 1.053     | 0.001 |10.595 |
|Simple       |   Yes   | 0.704 | 0.993     | 0.000 |19.764 |
|ResNet18     |   Yes   | 0.685 | 1.027     | 0.001 |12.665 |
-->
# NemodriveFinal
