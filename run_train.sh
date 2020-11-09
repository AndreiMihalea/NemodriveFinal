#!/bin/bash

NUM_EPOCHS=20
STEP_SIZE=20
BATCH_SIZE=128
LR=0.00001
WEIGHT_DECAY=0.0001
OPTIMIZER=rmsprop

VIS_INT=100
LOG_INT=50
DATASET_DIR=./dataset
MODEL=resnet

#export CUDA_VISIBLE_DEVICES=1

# train model using 2D persepctive augmentation and append speed
echo $MODEL" + SPEED + BALANCE + AUG"
python3.6 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_speed \
	--use_aug \
	--use_balance \
	--dataset_dir $DATASET_DIR \
	--step_size $STEP_SIZE \
	--num_epochs $NUM_EPOCHS \
	--optimizer $OPTIMIZER \
	--weight_decay $WEIGHT_DECAY\
	--lr $LR \
	--use_old
