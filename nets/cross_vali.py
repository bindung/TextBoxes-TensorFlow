import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os, os.path
import sys
sys.path.insert(0,'../processing/')
sys.path.insert(0,'../')
from nets import txtbox_300, textbox_common, np_methods,txtbox512
#from processing import image_processing
from image_processing2 import *
from processing import ssd_vgg_preprocessing, visualization,txt_preprocessing
slim = tf.contrib.slim
from datasets import sythtextprovider
import numpy as np



match_threshold = [0.5]
min_scala= np.linspace(start=0.08,stop=0.16,num=9)
max_scala= np.linspace(start=0.6,stop=0.9,num=7)
thres = []
ran = []
sum_error = []
for min_s in min_scala:
	for max_s in max_scala:
		scales = [min_s + i*(max_s - min_s)/6  for i in range(7)]
		anchor_sizes = [(512*scales[i], 512*scales[i] + 50) for i in range(7)]
		with tf.Graph().as_default(): 
			# build a net
			params = txtbox512.TextboxNet.default_params
			params = params._replace(anchor_sizes = anchor_sizes)
			text_net = txtbox512.TextboxNet(params)
			text_shape = text_net.params.img_shape
			print 'text_shape '+  str(text_shape)
			text_anchors = text_net.anchors(text_shape)
			
			## dataset provider
			dataset = sythtextprovider.get_datasets('../data/ICDAR2013/',file_pattern='*.tfrecord')
			
			data_provider = slim.dataset_data_provider.DatasetDataProvider(
					dataset, common_queue_capacity=32, common_queue_min=2)
			
			[image, shape, glabels, gbboxes] = \
			data_provider.get(['image', 'shape',
							 'object/label',
							 'object/bbox'])
			
			dst_image, glabels, gbboxes,num = \
			txt_preprocessing.preprocess_image(image,  glabels,gbboxes, 
													text_shape,is_training=True)

			glocalisations, gscores = \
			text_net.bboxes_encode( gbboxes, text_anchors, num)
			for i in range(6):
				glocalisations[i] = tf.expand_dims(glocalisations[i], axis=0)
				gscores[i] = tf.expand_dims(gscores[i], axis=0)
			
			with tf.Session() as sess: 
				sess.run(tf.global_variables_initializer())
				with slim.queues.QueueRunners(sess):
					error = []
					for i in xrange(500):
						rpredictions, rlocalisations,gbboxes_= sess.run([gscores, glocalisations,gbboxes])
						#rpredictions_2 = list(rpredictions)
						#localb = []
						for i in range(6):
							#decodeb = np_methods.ssd_bboxes_decode(rlocalisations[i],text_anchors[i])
							#localb.append(decodeb[np.where(rpredictions[i] > 0.2)])
							pre2 = np.expand_dims(1-rpredictions[i], -1)
							rpredictions[i] = np.concatenate([pre2, np.expand_dims(rpredictions[i], -1)],axis = -1)
						rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
								rpredictions, rlocalisations, text_anchors,
								select_threshold=0.001, img_shape=text_shape, num_classes=2, decode=True)

						rbboxes = np_methods.bboxes_clip(rbboxes)
						rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=-1)
						rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, 
																		  nms_threshold=0.45)
						#Resize bboxes to original image shape. Note: useless for Resize.WARP!
						#bboxes = np.concatenate(localb, 0)

						error.append((gbboxes_.shape[0] - rbboxes.shape[0]))
					print "THe match_threshold is %s with range %s, error is %s" %(min_s, max_s, sum(error))
					sum_error.append(sum(error))
					thres.append(min_s)
					ran.append(max_s)
result = np.stack([sum_error, thres, ran]).T
np.savetxt(fname='result.csv',X=result, delimiter=',',fmt='%.2f')
