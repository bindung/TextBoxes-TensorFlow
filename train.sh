#/bin/bash

DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/train_NHWC
CUDA_VISIBLE_DEVICES=0,1 python Textbox_train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=momentum \
    --learning_rate=0.001 \
    --batch_size=8 \
    --num_samples=800000 \
    --gpu_memory_fraction=0.95 \
    --max_number_of_steps=500000 \
    --use_batch=False \
    --num_clones=2
