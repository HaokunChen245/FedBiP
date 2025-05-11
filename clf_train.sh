#!/bin/bash
idx=0
NUM_SHOTS=(80)
DATASETS=("domainnet")
for DATASET in "${DATASETS[@]}"
do
    for N_SHOT in "${NUM_SHOTS[@]}"
    do
        TRAIN_TYPE="train_syn_wnoise_0.1_interpolated_${N_SHOT}"
        CUDA_VISIBLE_DEVICES=$idx python clf_train.py \
            --train_type $TRAIN_TYPE \
            --dataset $DATASET \
            --pretrained \
            > logs_${DATASET}/log_${TRAIN_TYPE}.txt &
        idx=$((idx+1))
    done
done
