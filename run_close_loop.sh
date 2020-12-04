#!/bin/bash

######################################################################
################ OLD DATASET #########################################
#####################################################################

BEGIN=0
END=100
MODEL=resnet
LOAD_MODEL=00000
SPLIT_PATH=data_split/old_dataset/rand_split/test_scenes.txt
DATA_PATH=/home/nemodrive/workspace/roberts/UPB_dataset/old_dataset
SIM_DIR=simulation

python close_loop.py \
	--begin $BEGIN \
	--end $END \
	--model $MODEL \
	--load_model $LOAD_MODEL \
	--split_path $SPLIT_PATH \
	--data_path $DATA_PATH \
	--sim_dir $SIM_DIR \
	--use_speed \
	--use_old \
	
###################################################################
####################### NEW DATASET ###############################
###################################################################

# BEGIN=0
# END=10
# MODEL=resnet
# LOAD_MODEL=resnet_speed_augm_old_balance
# SPLIT_PATH=data_split/new_dataset/rand_split/test_scenes.txt
# DATA_PATH=/home/nemodrive/workspace/roberts/UPB_dataset/new_dataset
# SIM_DIR=simulation

# python close_loop.py \
#	--begin $BEGIN \
# 	--end $END \
#	--model $MODEL \
# 	--load_model $LOAD_MODEL \
#	--split_path $SPLIT_PATH \
#	--data_path $DATA_PATH \
# 	--sim_dir $SIM_DIR \
# 	--use_speed \
