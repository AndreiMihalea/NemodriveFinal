#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################
ROOT_DIR=/home/nemodrive/workspace/roberts/UPB_dataset/all_10fps

# for the old dataset
#python -m create_dataset.create_dataset \
# --root_dir $ROOT_DIR\
# --use_pose\
# --frame_rate 10\ 

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

## for the old dataset
#python -m create_dataset.split_dataset \
# --train data_split/train_scenes.txt \
# --test data_split/test_scenes.txt \
# --use_pose \


########################################################################
############### COMPUTE DATASET WEIGHTS ################################
########################################################################
## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 32.8 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 22.8 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 25.3 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 27.8 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 30.3 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 35.3 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 37.8 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 40.3 \

## for the old dataset
python -m create_dataset.weights \
 --use_pose \
 --scale 42.8 \

