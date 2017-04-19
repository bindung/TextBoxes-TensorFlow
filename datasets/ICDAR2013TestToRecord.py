##This program change the ground truth file in ICDAR test to tfrecord

import numpy as np 
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf 
import re
from datasets_test.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
from PIL import Image

# The path of the ground truth file and image
# Change this path to the directory of the ground truth txts
ground_truth_path = '/Users/kiyoshi/Desktop/Research/LAB/FirstAccessment/2ndPaper/Textbox-Tensor/ICDAR-Test-GT/'
# Change this path to the directory of the image files
image_path = '/Users/kiyoshi/Desktop/Research/LAB/FirstAccessment/2ndPaper/Textbox-Tensor/ICDAR-Test-Images/'

# Read from and parse the txt files
def readGT(gt_dir):
	#create structure for the columns
	gt_coordinate_and_words = []
	gt_names = []
	path_list = []
	txt_name_list = []

	# Save the paths for all gound truth txt files
	for txt_name in os.listdir(gt_dir):
		txt_path = ground_truth_path + txt_name
		txt_name_list.append(txt_name)
		path_list.append(txt_path)
	#print path_list
	
	# Only include the txt paths with right contents
	"""
	for path in path_list:
		file_content = open(path,'rU').readlines()
		number_of_elements = 0
		for lines in file_content:
			lines = lines.strip()
     		elements_in_a_line = lines.split(",")    		
     		number_of_elements += len(elements_in_a_line)

     	if number_of_elements % 5 == 0:
 			true_path_list.append(path)
 			
 	print true_path_list
	"""

	for file_path in path_list:
		try:
			gt_file = np.loadtxt(file_path, dtype={'names':('xmin','ymin','xmax','ymax','word'),'formats':(np.float, np.float, np.float, np.float, '|S15')}, delimiter = ',')
			gt_coordinate_and_words.append(gt_file)
			#true_path_list.append(file_path)
			
			# Save image names
			#for image_name in txt_name_list: 
			image_name = os.path.basename(file_path)
			if len(image_name) == 12:
				imname = image_name[3:8] + '.jpg'
			if len(image_name) == 13:
				imname = image_name[3:9] + '.jpg'
			if len(image_name) == 14:
				imname = image_name[3:10] +'.jpg'
			gt_names.append(imname)
		except ValueError:
			continue
	#print gt_coordinate_and_words[1][:]
	#print gt_names
				
	return gt_names, gt_coordinate_and_words

def _convert_to_example(image_data, shape, bbox, label, imname):
	nbbox = np.array(bbox)
	ymin = list(nbbox[:, 0])
	xmin = list(nbbox[:, 1])
	ymax = list(nbbox[:, 2])
	xmax = list(nbbox[:, 3])

	"""
	print ymin
	print xmin
	print ymax
	print xmax
	"""
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
	#print example
	return example


# Deal with the image and the labels
def _image_processing(wordbb, imname, coder):
	# Read image according to the imname
	imname_path = image_path + imname
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
	gt_names, gt_coordinate_and_words = readGT(ground_truth_path)
	coder = ImageCoder()
	tf_filename = 'ICDAR2013_Test.tfrecord'
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
	print 'Transform to tfrecord finished!'
	print 'The size of testing data is ' + str(len(gt_names))

if __name__ == '__main__':
	main()
	
