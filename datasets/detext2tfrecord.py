# Transfer DeText validation data into tfrecord

import numpy as np 
import scipy.io as sio
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import re
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
from PIL import Image

data_path = '../data/DeTEXT/DeTEXT_validation/'
ground_truth_path = '../data/DeTEXT/DeTEXT_validation_gt/'

# Read from the groud truth file and parse
def readGT(gt_dir):
	gt_coordinate = []
	gt_names = []
	path_list = []
	txt_list = []

	for txt_name in os.listdir(gt_dir):
		txt_path = str(ground_truth_path) + txt_name
		txt_list.append(txt_name)
		path_list.append(txt_path)
	#print len(path_list)

	for file_path in path_list:
		try:
			gt_file = np.loadtxt(file_path, dtype={'names':('x1','y1','x2','y2','x3','y3','x4','y4','label'),'formats':(np.float, np.float, np.float, np.float,np.float,np.float,np.float,np.float, '|S100')}, delimiter = ',')
			#gt_file = np.genfromtxt(file_path, delimiter=',', dtype=None, names=('x1', 'y1', 'x2', 'y2', 'x3','y3','x4','y4','label'))
			gt_coordinate.append(gt_file)
			# Save image names
			image_name = os.path.basename(file_path)
			pattern = re.compile(r"\d+")
			imname = 'img_' + pattern.findall(image_name)[0] + '.jpg'
			gt_names.append(imname)
		except ValueError:
			pass	
	
	print len(gt_names)
	return gt_names, gt_coordinate


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
	imname_path = data_path + imname
	
	image_data = tf.gfile.GFile(imname_path, 'r').read()
	image = coder.decode_jpeg(image_data)
	shape = image.shape
	
	# The number of boxes in an image
	bbox = []

	ymin = wordbb['y3']
	xmin = wordbb['x1']
	ymax = wordbb['y1']
	xmax = wordbb['x2']

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
	gt_names, gt_coordinate = readGT(ground_truth_path)
	tf_filename = '../DeTEXT.tfrecord'
	tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
	
	# Generate index and shuffle
	index = [i for i in range(len(gt_names))]
	random_index = np.random.permutation(index)
	
	# Deal with every image
	for i in random_index:
		imname = gt_names[i]
		wordbb = gt_coordinate[i]
		image_data, shape, bbox, label, imname = _image_processing(wordbb, imname, coder)
		print imname
		example = _convert_to_example(image_data, shape, bbox, label, imname)
		tfrecord_writer.write(example.SerializeToString())

	print 'Transform test set to tfrecord finished!'

if __name__ == '__main__':
	main()
