import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os, os.path
import sys
sys.path.insert(0,'../processing/')
sys.path.insert(0,'../')
from nets import txtbox_300, textbox_common, np_methods
#from processing import image_processing
from image_processing2 import *
from processing import ssd_vgg_preprocessing, visualization,txt_preprocessing
slim = tf.contrib.slim
from datasets import sythtextprovider
import numpy as np



match_threshold = [0.5]
scales_range = [0.05,0.1,0.15,0.2]
thres = []
ran = []
sum_error = []
for threshold in match_threshold:
	for s in scales_range:
		scales = [s + i*(0.6 - s)/5  for i in range(6)]
		with tf.Graph().as_default(): 
			# build a net
			params = txtbox_300.TextboxNet.default_params
			params = params._replace( match_threshold=threshold)
			params = params._replace( scales=scales)
			text_net = txtbox_300.TextboxNet(params)
			text_shape = text_net.params.img_shape
			print 'text_shape '+  str(text_shape)
			text_anchors = text_net.anchors(text_shape)
			print text_net.params.match_threshold
			
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
					for i in xrange(229):
						rpredictions, rlocalisations, img ,gbboxes_= sess.run([gscores, glocalisations,dst_image,gbboxes])
						rpredictions_2 = list(rpredictions)
						localb = []
						for i in range(6):
							decodeb = np_methods.ssd_bboxes_decode(rlocalisations[i],text_anchors[i])
							localb.append(decodeb[np.where(rpredictions[i] > 0.2)])
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
						bboxes = np.concatenate(localb, 0)
						image_ = np.uint8(img)*255
						'''
						img = image_.copy()
						visualize_bbox(img, rbboxes)
						img = image_.copy()
						visualize_bbox(img, bboxes)
						img = image_.copy()
						visualize_bbox(img, gbboxes_)
						'''
						error.append((gbboxes_.shape[0] - rbboxes.shape[0]))
					print "THe match_threshold is %s with range %s, error is %s" %(threshold, s, sum(error))
					sum_error.append(sum(error))
					thres.append(threshold)
					ran.append(s)
result = np.stack([sum_error, thres, ran]).T
np.savetxt(fname='result.csv',X=result, delimiter=',',fmt='%.4f')
