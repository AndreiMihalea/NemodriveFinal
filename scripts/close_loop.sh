#!/bin/bash

######################################################################
################ OLD DATASET #########################################
#####################################################################

BEGIN=0
END=2
MODEL=resnet
LOAD_MODEL=00002
SPLIT_PATH=data_split/test_scenes.txt
DATA_PATH=/mnt/storage/workspace/andreim/nemodrive/UPB_dataset_robert/old_dataset
SIM_DIR=simulation

python close_loop.py \
	--begin $BEGIN \
	--end $END \
	--load_model $LOAD_MODEL \
	--split_path $SPLIT_PATH \
	--data_path $DATA_PATH \
	--sim_dir $SIM_DIR \
	--model deeplabv3 \
	--backbone resnet50 \
	--dataset upb
