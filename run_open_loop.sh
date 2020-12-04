#!/bin/bash

BATCH_SIZE=128
DATASET_DIR=./dataset
LOAD_MODEL=00000

python open_loop.py \
	--batch_size $BATCH_SIZE \
	--dataset_dir $DATASET_DIR \
	--load_model $LOAD_MODEL \
	--use_speed \
#	--use_old \
