#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################

## for the old dataset
#python3.6 create_dataset.py \
#  --root_dir /home/robert/PycharmProjects/upb_dataset \
#  --use_old \

## for the new dataset
#python3.6 create_dataset.py \
#  --root_dir /home/robert/PycharmProjects/upb_dataset_new \

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

## for the old dataset
#python3.6 split_dataset.py \
#  --train ../data_split/old_dataset/geo_split/train_scenes.txt \
#  --test ../data_split/old_dataset/geo_split/test_scenes.txt \
#  --use_old

## for the new dataset
#python3.6 split_dataset.py \
#  --train ../data_split/new_dataset/rand_split/train_scenes.txt \
#  --test ../data_split/new_dataset/rand_split/test_scenes.txt \


########################################################################
############## CREATE THE AUGMENTATION DATASET #########################
########################################################################

## for the old dataset
#python3.6 create_aug_dataset.py \
#  --root_dir /home/robert/PycharmProjects/upb_dataset \
#  --train /home/robert/PycharmProjects/NemodriveFinal/data_split/old_dataset/geo_split/train_scenes.txt \
#  --use_old

## for the new dataset
#python3.6 create_aug_dataset.py \
#  --root_dir /home/robert/PycharmProjects/upb_dataset_new \
#  --train /home/robert/PycharmProjects/NemodriveFinal/data_split/new_dataset/rand_split/train_scenes.txt \


#######################################################################
############## COMPUTE DATASET WEIGHTS ################################
#######################################################################

# for the old dataset
python3.6 weights.py \
  --augm \
  --use_old \

## for the new dataset
#python3.6 weights.py \
#  --augm

