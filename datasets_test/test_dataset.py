# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This program change the dataset(which includs 4262 images) into tfRecord format.

"""

import os
import sys
import random

import numpy as np
import tensorflow as tf
import scipy.io as sio

#from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
#from datasets.pascalvoc_common import VOC_LABELS

#DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPGImages/'

def process_image(directory,name):

	#Read images
	filename = directory + DIRECTORY_IMAGES + name + '.jpg'
	image_data = tf.gfile.FastGFile(filename,'r').read()

	#Generate label files
	 