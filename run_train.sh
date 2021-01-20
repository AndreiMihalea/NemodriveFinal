#!/bin/bash

NUM_EPOCHS=20
STEP_SIZE=5
PATIENCE=15
BATCH_SIZE=256
LR=0.01
FINAL_LR=0.0001
WEIGHT_DECAY=0.01
OPTIMIZER=sgd

VIS_INT=100
LOG_INT=50
DATASET_DIR=./dataset

export CUDA_VISIBLE_DEVICES=1

# train model using 2D persepctive augmentation and append speed
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
	--final_lr $FINAL_LR\
	--seed 0\
	--use_balance \
	--use_augm \
