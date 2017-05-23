

INTERVAL=120


#GPU-0
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-32130
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g1
CUDA_VISIBLE_DEVICES=0 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \
    --use_batch=True 

CHECKPOINT_PATH=./logs/ICDAR2013/g1
EVAL_DIR=./logs/evals/g1
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=0 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################


#GPU-1
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-32130
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g2
CUDA_VISIBLE_DEVICES=1 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \


CHECKPOINT_PATH=./logs/ICDAR2013/g2
EVAL_DIR=./logs/evals/g2
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=1 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################

#GPU-2
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-805859
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g3
CUDA_VISIBLE_DEVICES=2 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \


CHECKPOINT_PATH=./logs/ICDAR2013/g3
EVAL_DIR=./logs/evals/g3
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=2 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################


#GPU-3
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-11555
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g4
CUDA_VISIBLE_DEVICES=3 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \
    --use_hard_neg=True


CHECKPOINT_PATH=./logs/ICDAR2013/g4
EVAL_DIR=./logs/evals/g4
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=3 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################


#GPU-4
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-11555
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g5
CUDA_VISIBLE_DEVICES=4 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \



CHECKPOINT_PATH=./logs/ICDAR2013/g5
EVAL_DIR=./logs/evals/g5
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=4 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################



#GPU-5
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-12616
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g6
CUDA_VISIBLE_DEVICES=5 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \
	--use_hard_neg=True


CHECKPOINT_PATH=./logs/ICDAR2013/g6
EVAL_DIR=./logs/evals/g6
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=5 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################

#GPU-6
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-12616
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g7
CUDA_VISIBLE_DEVICES=6 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \



CHECKPOINT_PATH=./logs/ICDAR2013/g7
EVAL_DIR=./logs/evals/g7
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=6 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################


#GPU-7
###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-18418
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g8
CUDA_VISIBLE_DEVICES=7 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=${INTERVAL} \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.00001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.8 \
	--max_number_of_steps=5000000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \
    --use_batch=True



CHECKPOINT_PATH=./logs/ICDAR2013/g8
EVAL_DIR=./logs/evals/g8
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=7 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.06

###################################







