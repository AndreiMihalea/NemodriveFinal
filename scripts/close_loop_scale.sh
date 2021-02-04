#!/bin/bash

######################################################################
################ OLD DATASET #########################################
#####################################################################

BEGIN=0
END=100
MODEL=resnet
SPLIT_PATH=data_split/test_scenes.txt
DATA_PATH=/home/nemodrive/workspace/roberts/UPB_dataset/old_dataset
SIM_DIR=simulation_scale


for i in $(seq 5 7); do
	python close_loop.py \
		--begin $BEGIN \
		--end $END \
		--load_model 0000$i \
		--split_path $SPLIT_PATH \
		--data_path $DATA_PATH \
		--sim_dir $SIM_DIR 
done
