#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################

#ROOT_DIR=/home/nemodrive/workspace/roberts/UPB_dataset/old_dataset
#ROOT_DIR=/home/nemodrive/workspace/roberts/UPB_dataset/all_10fps
ROOT_DIR=/home/robert/PycharmProjects/upb_dataset

## for the old dataset
python -m scripts.create_dataset \
 --root_dir $ROOT_DIR\
 --frame_rate 3\
 --use_pose \

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

### for the old dataset
#python -m scripts.split_dataset \
# --train data_split/train_scenes.txt \
# --test data_split/test_scenes.txt \
#
########################################################################
############### COMPUTE DATASET WEIGHTS ################################
########################################################################
#
### for the old dataset
#python -m scripts.weights

