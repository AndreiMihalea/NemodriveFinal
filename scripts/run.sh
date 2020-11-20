#!/bin/bash

################################################################################
############################# CREATE DATASET WITHOUT AUGMENTATION ##############
################################################################################

# for the old dataset
# python create_dataset.py \
#  --root_dir /home/nemodrive/workspace/roberts/UPB_dataset/old_dataset \
#  --use_old \

# for the new dataset
# python create_dataset.py \
#  --root_dir /home/nemodrive/workspace/roberts/UPB_dataset/new_dataset \

########################################################################
#################### SPLIT THE DATASET #################################
########################################################################

# for the old dataset
# python split_dataset.py \
#  --train ../data_split/old_dataset/rand_split/train_scenes.txt \
#  --test ../data_split/old_dataset/rand_split/test_scenes.txt \
#  --use_old

# for the new dataset
# python split_dataset.py \
#  --train ../data_split/new_dataset/rand_split/train_scenes.txt \
#  --test ../data_split/new_dataset/rand_split/test_scenes.txt \


########################################################################
############## CREATE THE AUGMENTATION DATASET #########################
########################################################################

## for the old dataset
# python create_aug_dataset.py \
#  --root_dir /home/nemodrive/workspace/roberts/UPB_dataset/old_dataset \
#  --train ../data_split/old_dataset/geo_split/train_scenes.txt \
#  --use_old

# for the new dataset
# python create_aug_dataset.py \
#  --root_dir /home/nemodrive/workspace/roberts/UPB_dataset/new_dataset \
#   --train ../data_split/new_dataset/rand_split/train_scenes.txt \


#######################################################################
############## COMPUTE DATASET WEIGHTS ################################
#######################################################################

# for the old dataset
#python weights.py \
#  --augm \
#  --use_old \

# for the new dataset
#  python weights.py \
#  --augm

