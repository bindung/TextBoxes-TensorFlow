cd /media/MMVCNYLOCAL/MMVC_NY/David_jin/TextBoxes-TensorFlow/

DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/


在全数据集上不用hard negative, 
426 预测良好， log_train ./logs_426/model.ckpt-48406
     
DATASET_DIR=./data/ICDAR2013/  


CHECKPOINT_PATH=./logs_426/model.ckpt-48406

CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
CHECKPOINT_PATH=./checkpoints/model.ckpt-13889
CHECKPOINT_PATH=./logs/momentum_0.001/model.ckpt-21218
DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/train/test_runtime
TF_ENABLE_WINOGRAD_NONFUSED=1 CUDA_VISIBLE_DEVICES=4,5,6,7 setsid python Textbox_train.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=3600 \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.001 \
	--batch_size=8 \
	--match_threshold=0.5 \
	--num_samples=3200000 \
	--gpu_memory_fraction=0.95 \
	--max_number_of_steps=600 \
    --use_batch=False \
	--num_clones=4 \
    --use_batch=False
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=text_box_300 \
    --ignore_missing_vars=True \
    --use_batch=True
    



CHECKPOINT_PATH=./logs/train/logs609
EVAL_DIR=./logs/eval/logs609
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=0 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --use_batch=False \
    --gpu_memory_fraction=0.02



###########################################
CHECKPOINT_PATH=./checkpoints/model.ckpt-32130
DATASET_DIR=./data/ICDAR2013/train
TRAIN_DIR=./logs/ICDAR2013/g1
CUDA_VISIBLE_DEVICES=0 setsid python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=300 \
	--save_interval_secs=3600 \
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
    --use_batch=True \
    --use_hard_neg=True

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

mo
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
CHECKPOINT_PATH=./checkpoints/model.ckpt-12325
DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/
python Textbox_train.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=600 \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.004 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--gpu_train=/cpu:0 \
	--gpu_memory_fraction=0.95 \
	--max_number_of_steps=400000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --ignore_missing_vars=True \
\

CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/
python Train_single_gpu.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=600 \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.001 \
	--loss_alpha=1.0 \
	--batch_size=1 \
	--gpu_train=/cpu:0 \
	--gpu_memory_fraction=0.95 \
	--max_number_of_steps=400000 \
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --ignore_missing_vars=True 




python download_and_convert_data.py \
   	 	--dataset_name=cifar10 \
    	--dataset_dir="${DATA_DIR}"


DATASET_DIR=datasets/cifar10
TRAIN_DIR=train_logs
CUDA_VISIBLE_DEVICES=5,6 setsid  python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=vgg_16 \
    --num_clones=8

python convert.py \
	'/Users/xiaodiu/Documents/github/projecttextbox/caffe-tensorflow/examples/textbox/deploy.prototxt' \
	--caffemodel='/Users/xiaodiu/Documents/github/projecttextbox/caffe-tensorflow/examples/textbox/TextBoxes_icdar13.caffemodel' \
	--data-output-path='/Users/xiaodiu/Documents/github/projecttextbox/caffe-tensorflow/examples/textbox' \
	--code-output-path='/Users/xiaodiu/Documents/github/projecttextbox/caffe-tensorflow/examples/textbox' \
	-p=train


    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')

		gradient_multipliers = {
    		'conv1/conv1_1/weights_1' : 1.,
    		'conv1/conv1_1/biases_1' : 1.,
    		'conv1/conv1_2/weights_1' : 1.,
    		'conv1/conv1_2/biases_1' : 1.,
    		'conv2/conv2_1/weights_1' : 1.,
    		'conv2/conv2_1/biases_1' : 1.,
    		'conv2/conv2_2/weights_1' : 1.,
    		'conv2/conv2_2/biases_1' : 1.,
    		'conv3/conv3_1/weights_1' : 1.,
    		'conv3/conv3_1/biases_1' : 2.,
   			'conv3/conv3_2/weights_1' : 1.,
    		'conv3/conv3_2/biases_1' : 2.,
    		'conv4/conv4_1/weights_1' : 1.,
    		'conv4/conv4_1/biases_1' : 2.,
    		'conv4/conv4_2/weights_1' : 1.,
    		'conv4/conv4_2/biases_1' : 2.,
    		'conv4/conv4_3/weights_1' : 1.,
    		'conv4/conv4_3/biases_1' : 2.,
    		'conv5/conv5_1/weights_1' : 1.,
    		'conv5/conv5_1/biases_1' : 2.,
    		'conv5/conv5_2/weights_1' : 1.,
    		'conv5/conv5_2/biases_1' : 2.,
    		'conv5/conv5_3/weights_1' : 1.,
    		'conv5/conv5_3/biases_1' : 2.,
    		'conv6/weights_1' : 1.,
    		'conv6/biases_1' : 2.,
    		'conv7/weights_1' : 1.,
    		'conv7/biases_1' : 2.,
    		'conv8/conv1x1/weights_1' :1.,
    		'conv8/conv1x1/biases_1' :2.,
    		'conv8/conv3x3/weights_1' :1.,
    		'conv8/conv3x3/biases_1' :2.,
    		'conv9/conv1x1/weights_1' :1.,
    		'conv9/conv1x1/biases_1' :2.,
    		'conv9/conv3x3/weights_1' :1.,
    		'conv9/conv3x3/biases_1' :2.,
    		'conv10/conv1x1/weights_1' :1.,
    		'conv10/conv1x1/biases_1' :2.,
    		'conv10/conv3x3/weights_1' :1.,
    		'conv10/conv3x3/biases_1' :2.,
    		'global/conv1x1/weights_1' :1.,
    		'global/conv1x1/biases_1' :2.,
    		'global/conv3x3/weights_1' :1.,
    		'global/conv3x3/biases_1' :2.,
    		'conv4_box/conv_cls/weights_1': 1.,
    		'conv4_box/conv_cls/biases_1': 2.,
    		'conv4_box/conv_loc/weights_1': 1.,
    		'conv4_box/conv_loc/biases_1': 2.,
    		'conv7_box/conv_loc/weights_1': 1.,
    		'conv7_box/conv_loc/biases_1': 2.,
    		'conv7_box/conv_cls/weights_1': 1.,
    		'conv7_box/conv_cls/biases_1': 2.,
    		'conv8_box/conv_loc/weights_1': 1.,
    		'conv8_box/conv_loc/biases_1': 2.,
    		'conv8_box/conv_cls/weights_1': 1.,
    		'conv8_box/conv_cls/biases_1': 2.,
    		'conv9_box/conv_loc/weights_1': 1.,
    		'conv9_box/conv_loc/biases_1': 2.,
    		'conv9_box/conv_cls/weights_1': 1.,
    		'conv9_box/conv_cls/biases_1': 2.,
    		'conv10_box/conv_cls/weights_1': 1.,
    		'conv10_box/conv_cls/biases_1': 2.,
    		'conv10_box/conv_loc/weights_1': 1.,
    		'conv10_box/conv_loc/biases_1': 2.,
    		'global_box/conv_cls/weights_1': 1.,
    		'global_box/conv_cls/biases_1': 2.,
    		'global_box/conv_loc/weights_1': 1.,
    		'global_box/conv_loc/biases_1': 2.
  		}