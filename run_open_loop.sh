#!/bin/bash

BATCH_SIZE=128
DATASET_DIR=./dataset
LOAD_MODEL=00002

python open_loop.py \
	--batch_size $BATCH_SIZE \
	--dataset_dir $DATASET_DIR \
	--load_model $LOAD_MODEL \
