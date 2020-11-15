#!/bin/bash

BATCH_SIZE=128
DATASET_DIR=./dataset
LOAD_MODEL=resnet_speed_augm_old_balance

python open_loop.py \
	--batch_size $BATCH_SIZE \
	--dataset_dir $DATASET_DIR \
	--load_model $LOAD_MODEL \
	--use_speed \
	--use_old \
