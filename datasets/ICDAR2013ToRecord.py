##This program change the ground truth file in ICDAR test to tfrecord

import numpy as np 
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf 
import re
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
from PIL import Image

data_path = 'data/ICDAR2013/'

tf.app.flags.DEFINE_string(
 'dataset', 'train',
 'the dataset is for training or testing')

tf.app.flags.DEFINE_string(
 'ground_truth_path_test', '../data/ICDAR2013/ICDAR-Test-GT/',
 'Directory of the ground truth txt for test set .')

tf.app.flags.DEFINE_string(
 'ground_truth_path_train', '../data/ICDAR2013/ICDAR-Training-GT/',
 'Directory of the ground truth txt for training set .')

tf.app.flags.DEFINE_string(
 'image_path_test', '../data/ICDAR2013/ICDAR-Test-Images/',
 'Directory of ICDAR2013 test data.')

tf.app.flags.DEFINE_string(
 'image_path_train', '../data/ICDAR2013/ICDAR-Training-Images/',
 'Directory of ICDAR2013 training data.')

tf.app.flags.DEFINE_string(
 'tf_filename_test', '../data/ICDAR2013/ICDAR2013_Test.tfrecord',
 'test set tfrecord file name')

tf.app.flags.DEFINE_string(
 'tf_filename_train', '../data/ICDAR2013/ICDAR2013_Train.tfrecord',
 'train set tfrecord file name')

FLAGS = tf.app.flags.FLAGS

arg = str(sys.argv[1:])
"""
# The path of the ground truth file and image
ground_truth_path_test = '../data/ICDAR2013/ICDAR-Test-GT/'
image_path_test = '../data/ICDAR2013/ICDAR-Test-Images/'

ground_truth_path_train = '../data/ICDAR2013/ICDAR-Training-GT/'
image_path_train = '../data/ICDAR2013/ICDAR-Training-Images/'
"""

# Read from and parse the txt files
def readGT(gt_dir):
	#create structure for the columns
	gt_coordinate_and_words = []
	gt_names = []
	path_list = []
	txt_name_list = []

	if (arg.startswith("['--ground_truth_path_test")):
	# Save the paths for all gound truth txt files
		for txt_name in os.listdir(gt_dir):
			txt_path = str(FLAGS.ground_truth_path_test) + txt_name
			txt_name_list.append(txt_name)
			path_list.append(txt_path)

		for file_path in path_list:
			try:
				gt_file = np.loadtxt(file_path, dtype={'names':('xmin','ymin','xmax','ymax','word'),'formats':(np.float, np.float, np.float, np.float, '|S15')}, delimiter = ',')
				gt_coordinate_and_words.append(gt_file)
				
				# Save image names
				image_name = os.path.basename(file_path)
				"""
				if len(image_name) == 12:
					imname = image_name[3:8] + '.jpg'
				if len(image_name) == 13:
					imname = image_name[3:9] + '.jpg'
				if len(image_name) == 14:
					imname = image_name[3:10] +'.jpg'
				"""
				pattern = re.compile(r"\d+")
				imname = 'img_' + pattern.findall(image_name)[0] + '.jpg'
				gt_names.append(imname)

			except ValueError:
				pass

	else:
		for txt_name in os.listdir(gt_dir):
			txt_path = str(FLAGS.ground_truth_path_train) + txt_name
			txt_name_list.append(txt_name)
			path_list.append(txt_path)

		for file_path in path_list:
			try:
				gt_file = np.loadtxt(file_path, dtype={'names':('xmin','ymin','xmax','ymax','word'),'formats':(np.float, np.float, np.float, np.float, '|S15')}, delimiter = ' ')
				gt_coordinate_and_words.append(gt_file)

				# Save image names
				image_name = os.path.basename(file_path)
				pattern = re.compile(r"\d+")
				imname = pattern.findall(image_name)[0] + '.jpg'
				gt_names.append(imname)

			except ValueError:
				pass
		print gt_names
				
	return gt_names, gt_coordinate_and_words

def _convert_to_example(image_data, shape, bbox, label, imname):
	nbbox = np.array(bbox)
	ymin = list(nbbox[:, 0])
	xmin = list(nbbox[:, 1])
	ymax = list(nbbox[:, 2])
	xmax = list(nbbox[:, 3])

	print 'shape:{}, height:{}, width:{}'.format(shape, shape[0], shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(shape[0]),
			'image/width': int64_feature(shape[1]),
			'image/channels': int64_feature(shape[2]),
			'image/shape': int64_feature(shape),
			'image/object/bbox/ymin': float_feature(ymin),
			'image/object/bbox/xmin': float_feature(xmin),
			'image/object/bbox/ymax': float_feature(ymax),
			'image/object/bbox/xmax': float_feature(xmax),
			'image/object/bbox/label': int64_feature(label),
			'image/format': bytes_feature('jpeg'),
			'image/encoded': bytes_feature(image_data),
			'image/name': bytes_feature(imname),
			}))
	return example


# Deal with the image and the labels
def _image_processing(wordbb, imname, coder):
	# Read image according to the imname
	if (arg.startswith("['--ground_truth_path_test")): 
		imname_path = FLAGS.image_path_test + imname
	else:
		imname_path = FLAGS.image_path_train + imname
	
	image_data = tf.gfile.GFile(imname_path, 'r').read()
	image = coder.decode_jpeg(image_data)
	shape = image.shape
	
	# The number of boxes in an image
	bbox = []

	ymin = wordbb['ymin']
	xmin = wordbb['xmin']
	ymax = wordbb['ymax']
	xmax = wordbb['xmax']
	
	ymin = np.maximum(ymin/shape[0], 0.0)
	xmin = np.maximum(xmin/shape[1], 0.0)
	ymax = np.minimum(ymax/shape[0], 1.0)
	xmax = np.minimum(xmax/shape[1], 1.0)
	
	
	try:
		number_of_boxes = wordbb.shape[0]
	except IndexError:
		number_of_boxes = 1

	if (number_of_boxes == 1):
		bbox = [[ymin,xmin,ymax,xmax]]
	else:
		bbox = [[ymin[i],xmin[i],ymax[i],xmax[i]] for i in range(number_of_boxes)] 
	
	label = [1 for i in range(number_of_boxes)]
	shape = list(shape)

	#print bounding_box

	return image_data, shape, bbox, label, imname

def main():
	# Get gt_names and gt_coordinate_and_words
	coder = ImageCoder()

	if (arg.startswith("['--ground_truth_path_test")):
		gt_names, gt_coordinate_and_words = readGT(FLAGS.ground_truth_path_test)
		tf_filename = FLAGS.tf_filename_test
	else:
		gt_names, gt_coordinate_and_words = readGT(FLAGS.ground_truth_path_train)
		tf_filename = FLAGS.tf_filename_train

	tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
	# Generate index and shuffle
	index = [i for i in range(len(gt_names))]
	random_index = np.random.permutation(index)
	# Deal with every image
	for i in random_index:
		imname = gt_names[i]
		wordbb = gt_coordinate_and_words[i]
		#print wordbb
		image_data, shape, bbox, label, imname = _image_processing(wordbb, imname, coder)
		#print bounding_box
		print imname
		example = _convert_to_example(image_data, shape, bbox, label, imname)
		tfrecord_writer.write(example.SerializeToString())
		#print i

	if (arg.startswith("['--ground_truth_path_test")):
		print 'Transform test set to tfrecord finished!'
		print 'The size of data is ' + str(len(gt_names))
	else:
		print 'Transform training set to tfrecord finished!'
		print 'The size of data is ' + str(len(gt_names))

if __name__ == '__main__':
	main()
	
