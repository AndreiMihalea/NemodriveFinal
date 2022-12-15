#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################
ROOT_DIR=/home/nemodrive/workspace/roberts/UPB_dataset/all_10fps

# for the old dataset
#python -m create_dataset.create_dataset \
# --root_dir $ROOT_DIR\
# --frame_rate 10\

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

## for the old dataset
#python -m create_dataset.split_dataset \
# --train data_split/train_scenes.txt \
# --test data_split/test_scenes.txt \


########################################################################
############### COMPUTE DATASET WEIGHTS ################################
########################################################################

## for the old dataset
python -m create_dataset.weights \
