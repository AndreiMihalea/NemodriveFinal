#!/bin/bash

NUM_EPOCHS=5
STEP_SIZE=1
PATIENCE=$NUM_EPOCHS
BATCH_SIZE=256
LR=0.1
FINAL_LR=0.0001
WEIGHT_DECAY=0.001
OPTIMIZER=sgd
SCALE=32.8
SEED=0
ROI=input
ROI_MAP=seg_soft

VIS_INT=100
LOG_INT=25
VIS_DIR=snapshots_pose
LOG_DIR=logs_pose
DATASET_DIR=./dataset

export CUDA_VISIBLE_DEVICES=1


# train model using 2D perspectiv agumentation and data balancing
# this model is trained with the steering obtained from pose estimation
#python train.py \
#	--batch_size $BATCH_SIZE \
#	--vis_int $VIS_INT \
#	--log_int $LOG_INT \
#	--dataset_dir $DATASET_DIR \
#	--step_size $STEP_SIZE \
#	--patience $PATIENCE \
#	--num_epochs $NUM_EPOCHS \
#	--optimizer $OPTIMIZER \
#	--weight_decay $WEIGHT_DECAY \
#	--lr $LR \
#	--seed $SEED \
#	--scale $SCALE \
#	--vis_dir $VIS_DIR \
#	--log_dir $LOG_DIR \
#	--use_pose \



# train model using data balancing
# this model is trained with the steering obtained from pose estimation
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
	--scale $SCALE \
	--use_balance \
	--vis_dir $VIS_DIR \
	--log_dir $LOG_DIR \
	--use_pose \
	--use_roi $ROI \
	--roi_map $ROI_MAP \



# train model using 2D perspective augmentation
# this model is trained with the steering obtained from pose estimation
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
	--use_pose \
	--use_roi $ROI \
	--roi_map $ROI_MAP \


# train model using 2D perspective augmentation and data balancing
# this model is trained with the steering obtained from pose estimation
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
	--use_balance \
	--use_augm \
	--vis_dir $VIS_DIR \
	--log_dir $LOG_DIR \
	--use_pose \
	--use_roi $ROI \
	--roi_map $ROI_MAP \

