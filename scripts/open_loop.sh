#!/bin/bash

BATCH_SIZE=128
DATASET_DIR=./dataset
LOAD_MODEL=00003

python open_loop.py \
	--batch_size $BATCH_SIZE \
	--dataset_dir $DATASET_DIR \
	--load_model $LOAD_MODEL \
	--use_baseline \


#python open_loop.py \
#	--batch_size $BATCH_SIZE \
#	--dataset_dir $DATASET_DIR \
#	--load_model $LOAD_MODEL \
#	--use_baseline \
#	--use_synth \

