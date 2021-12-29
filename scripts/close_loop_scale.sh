#!/bin/bash

######################################################################
################ OLD DATASET #########################################
#####################################################################

BEGIN=77
END=100
MODEL=resnet
SPLIT_PATH=data_split/test_scenes.txt
DATA_PATH=/home/nemodrive/workspace/roberts/UPB_dataset/old_dataset
SIM_DIR=simulation_scale
EPOCH_CKPT=4

export CUDA_VISIBLE_DEVICES=1
for i in $(seq 7 7); do
	python close_loop.py \
		--begin $BEGIN \
		--end $END \
		--load_model 0000$i \
		--split_path $SPLIT_PATH \
		--data_path $DATA_PATH \
		--sim_dir $SIM_DIR \
    --epoch_ckpt $EPOCH_CKPT
done
