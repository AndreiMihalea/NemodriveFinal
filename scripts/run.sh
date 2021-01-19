#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################

## for the old dataset
python -m scripts.create_dataset \
 --root_dir /home/robert/PycharmProjects/upb_dataset\
 --frame_rate 3\

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

## for the old dataset
python -m scripts.split_dataset \
 --train data_split/train_scenes.txt \
 --test data_split/test_scenes.txt \

#######################################################################
############## COMPUTE DATASET WEIGHTS ################################
#######################################################################

## for the old dataset
python -m scripts.weights

