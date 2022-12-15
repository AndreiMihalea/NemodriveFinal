#!/bin/bash

NUM_EPOCHS=10
STEP_SIZE=1
PATIENCE=$NUM_EPOCHS
BATCH_SIZE=256
LR=0.1
WEIGHT_DECAY=0.001
OPTIMIZER=sgd

VIS_INT=100
LOG_INT=25
DATASET_DIR=./dataset
SEED=13
export CUDA_VISIBLE_DEVICES=1

# train model using 2D persepctive augmentation
# this model is trained with the steering recorded from CAN
# this model is trained on the raw dataset
#python train.py \
#	--batch_size $BATCH_SIZE \
#	--vis_int $VIS_INT \
#	--log_int $LOG_INT \
#	--dataset_dir $DATASET_DIR \
#	--step_size $STEP_SIZE \
#	--patience $PATIENCE \
#	--num_epochs $NUM_EPOCHS \
#	--optimizer $OPTIMIZER \
#	--weight_decay $WEIGHT_DECAY\
#	--lr $LR \
#	--seed $SEED \
#	--scale 1.0 \

# train model using 2D persepctive augmentation
# this model is trained with the steering recorded from CAN
# this model is trained with data balancing
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
	--seed $SEED \
	--use_balance \
 	--scale 1.0 \

# # train model using 2D persepctive augmentation
# # this model is trained with the steering recorded from CAN
# # this model is trained with databalancing and perspective augmentations
#python train.py \
# 	--batch_size $BATCH_SIZE \
# 	--vis_int $VIS_INT \
# 	--log_int $LOG_INT \
# 	--dataset_dir $DATASET_DIR \
# 	--step_size $STEP_SIZE \
# 	--patience $PATIENCE \
# 	--num_epochs $NUM_EPOCHS \
# 	--optimizer $OPTIMIZER \
# 	--weight_decay $WEIGHT_DECAY\
# 	--lr $LR \
# 	--seed $SEED \
# 	--use_augm \
# 	--scale 1.0 \


## train model using 2D persepctive augmentation
## this model is trained with the steering recorded from CAN
## this model is trained with data balancing, perspective augmentation
## and including a synthetic dataset during testing
#python train.py \
# 	--batch_size $BATCH_SIZE \
# 	--vis_int $VIS_INT \
# 	--log_int $LOG_INT \
# 	--dataset_dir $DATASET_DIR \
# 	--step_size $STEP_SIZE \
# 	--patience $PATIENCE \
# 	--num_epochs $NUM_EPOCHS \
# 	--optimizer $OPTIMIZER \
# 	--weight_decay $WEIGHT_DECAY\
# 	--lr $LR \
# 	--seed $SEED \
# 	--use_balance \
# 	--use_augm \
# 	--scale 1.0 \

