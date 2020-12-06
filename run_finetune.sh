#!/bin/bash

NUM_EPOCHS=100
STEP_SIZE=100
BATCH_SIZE=128
LR_FT=0.0001
WEIGHT_DECAY=0.01
OPTIMIZER=sgd

VIS_INT=50
LOG_INT=25
DATASET_DIR=./dataset
MODEL=00000

export CUDA_VISIBLE_DEVICES=0

# train model using 2D persepctive augmentation and append speed
echo $MODEL" + SPEED + BALANCE + AUG"
python train.py \
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
	--lr_ft $LR_FT \
	--load_model $MODEL \
	--finetune \
