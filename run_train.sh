#!/bin/bash

NUM_EPOCHS=100
STEP_SIZE=5
PATIENCE=15
BATCH_SIZE=128
LR=0.001
WEIGHT_DECAY=0.001
OPTIMIZER=sgd

VIS_INT=100
LOG_INT=50
DATASET_DIR=./dataset

export CUDA_VISIBLE_DEVICES=0

# train model using 2D persepctive augmentation and append speed
echo $MODEL" + SPEED + BALANCE + AUG"
python train.py \
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--dataset_dir $DATASET_DIR \
	--step_size $STEP_SIZE \
	--patience $PATIENCE \
	--num_epochs $NUM_EPOCHS \
	--optimizer $OPTIMIZER \
	--weight_decay $WEIGHT_DECAY\
	--lr $LR \
	--seed 0\
	--use_balance \
#	--use_augm \
#	--use_balance \
