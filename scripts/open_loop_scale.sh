#!/bin/bash

BATCH_SIZE=128
DATASET_DIR=./dataset

for i in $(seq 5 7); do
	python open_loop.py \
		--batch_size $BATCH_SIZE \
		--dataset_dir $DATASET_DIR \
		--load_model 0000$i 
done

