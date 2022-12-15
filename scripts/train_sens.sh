#!/bin/bash

NUM_EPOCHS=5
STEP_SIZE=1
PATIENCE=$NUM_EPOCHS
BATCH_SIZE=256
LR=0.1
FINAL_LR=0.0001
WEIGHT_DECAY=0.001
OPTIMIZER=sgd
SEED=0

VIS_INT=100
LOG_INT=25
VIS_DIR=snapshots_scale
LOG_DIR=logs_scale
DATASET_DIR=./dataset

export CUDA_VISIBLE_DEVICES=1


scales=(22.80 25.30 27.80 30.30 35.30 37.80 40.30 42.80)
# train model using 2D perspectiv agumentation and data balancing
# this model is trained with the steering obtained from pose estimation

for SCALE in ${scales[@]}; do
	echo $SCALE

	python train.py \
		--batch_size $BATCH_SIZE \
		--vis_int $VIS_INT \
		--log_int $LOG_INT \
		--dataset_dir $DATASET_DIR \
		--step_size $STEP_SIZE \
		--patience $PATIENCE \
		--num_epochs $NUM_EPOCHS \
		--optimizer $OPTIMIZER \
		--weight_decay $WEIGHT_DECAY \
		--lr $LR \
		--seed $SEED \
		--scale $SCALE\
		--use_augm \
		--vis_dir $VIS_DIR \
		--log_dir $LOG_DIR \
		--use_pose 
done
