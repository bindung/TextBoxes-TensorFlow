# TextBoxes-TensorFlow
TextBoxes re-implementation using tensorflow.
Much more info can be found in [Textbox](https://arxiv.org/pdf/1611.06779.pdf) and [SSD](https://arxiv.org/abs/1512.02325)
This project is greatly inspired by [slim project](https://github.com/tensorflow/models/tree/master/slim)  
And many functions are modified based on [SSD-tensorflow project](https://github.com/balancap/SSD-Tensorflow)  
Now the pipeline is much clear and can be resued in any tf projects.

Author: 
	Daitao Xing : dx383@nyu.edu
	Jin Huang   : jh5442@nyu.edu

# Progress
2017/ 03/14  

data_processing phase finished
Test：

	1. Download the dataset， put 1/ folder and gt.mat uner ddata/sythtext/ folder（will wirte script）   
	2. python datasets/data2record.py    
	3. python image_processing.py    
	
output： batch_size * 300 * 300 * 3 image

2017/ 03/17  

Finish the design of training(can start training)	

	TASET_DIR=./data/sythtext/
	TRAIN_DIR=./logs/
	python Textbox_train.py \
		--train_dir=${TRAIN_DIR} \
		--dataset_dir=${DATASET_DIR} \
		--save_summaries_secs=60 \
		--save_interval_secs=600 \
		--weight_decay=0.0005 \
		--optimizer=adam \
		--learning_rate=0.001 \
		--batch_size=2 \
	    	--gpu_data=/cpu:0 \
		--gpu_train=/cpu:0
		
2017/ 03/29

Overwrite all files, so make the training pipeline much clear.
	
	1. Write the load_batch . This can be resued in any preproceesing jobs.
	2. Rewrite the traning file, so make the pipeline more clear.
	

To generate tfrecords for both test and training sets for ICDAR2013 dataset:
```bash
cd datasets
```
Generate tfrecord for test set:
```bash
python ICDAR2013ToRecord.py --dataset=test --ground_truth_path_test=../data/ICDAR2013/ICDAR-Test-GT/
```
Generate tfrecord for training set:
```bash
python ICDAR2013ToRecord.py --dataset=train --ground_truth_path_train=../data/ICDAR2013/ICDAR-Training-GT/
```
The generated tfrecords are in TextBoxes-TensorFlow/data/ICDAR2013 folder.

# Problems to be solved： 
	1. The loss decreases slowly after 2000 iterations, find why?		
	2. Prepare the other two datasets(transform into tf.record)
	3. Evaluation part scripts.


