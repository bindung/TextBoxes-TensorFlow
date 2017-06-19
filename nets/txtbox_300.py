
""" 
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
Add descriptions
"""

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common

slim = tf.contrib.slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
TextboxParams = namedtuple('TextboxParameters', 
										['img_shape',
										 'num_classes',
										 'feat_layers',
										 'feat_shapes',
										 'scale_range',
										 'anchor_ratios',
										 'normalizations',
										 'prior_scaling',
										 'anchor_sizes',
										 'anchor_steps',
										 'scales',
										 'match_threshold'
										 ])

class TextboxNet(object):
	"""
	Implementation of the Textbox 300 network.

	The default features layers with 300x300 image input are:
	  conv4_3 ==> 38 x 38
	  fc7 ==> 19 x 19
	  conv6_2 ==> 10 x 10
	  conv7_2 ==> 5 x 5
	  conv8_2 ==> 3 x 3
	  pool6 ==> 1 x 1
	anchor_sizes=[(21., 45.),
		  (45., 99.),
		  (99., 153.),
		  (153., 207.),
		  (207., 261.),
		  (261., 315.)],

	anchor_sizes=[(30., 60.),
			  (60., 114.),
			  (114., 168.),
			  (168., 222.),
			  (222., 276.),
			  (276., 330.)],
	The default image size used to train this network is 300x300.
	"""
	default_params = TextboxParams(
		img_shape=(300, 300),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'global'],
		feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
		scale_range=[0.2, 0.95],
		anchor_ratios=[1.,2,3,5,7,10],
		normalizations=[20, -1, -1, -1, -1, -1],
		prior_scaling=[0.1, 0.1, 0.2, 0.2],
		anchor_sizes=[(30., 60.),
			  (60., 114.),
			  (114., 168.),
			  (168., 222.),
			  (222., 276.),
			  (276., 330.)],	
		anchor_steps=[8, 16, 30, 60, 100, 300],
		scales = [0.2 + i*0.8/5  for i in range(6)],
		#scales = [0.05, 0.1,0.15,0.25,0.4,0.65],
		match_threshold = 0.5
		)

	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params

	# ======================================================================= #
	def net(self, inputs,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_300',
			use_batch=False):
		"""
		Text network definition.
		"""
		r = text_net(inputs,
					feat_layers=self.params.feat_layers,
					normalizations=self.params.normalizations,
					is_training=is_training,
					dropout_keep_prob=dropout_keep_prob,
					reuse=reuse,
					use_batch=use_batch,
					scope=scope)

		return r

	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)


	def anchors(self, img_shape, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_common.textbox_achor_all_layers(img_shape,
									  self.params.feat_shapes,
									  self.params.anchor_ratios,
									  self.params.scales,
									  self.params.anchor_sizes,
									  0.5,
									  dtype)

	def bboxes_encode(self, bboxes, anchors, num,
					  scope='text_bboxes_encode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_text_bboxes_encode(
						bboxes, anchors, num,
						match_threshold=self.params.match_threshold,
						prior_scaling=self.params.prior_scaling,
						scope=scope)

	def bboxes_decode(self, feat_localizations, anchors, scope='ssd_bboxes_decode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_ssd_bboxes_decode(
			feat_localizations, anchors,
			prior_scaling=self.params.prior_scaling,
			scope=scope)

	def detected_bboxes(self, predictions, localisations,
						select_threshold=None, nms_threshold=0.5,
						clipping_bbox=None, top_k=400, keep_top_k=200):
		"""Get the detected bounding boxes from the SSD network output.
		"""
		# Select top_k bboxes from predictions, and clip
		rscores, rbboxes = \
			textbox_common.tf_ssd_bboxes_select(predictions, localisations,
											select_threshold=select_threshold,
											num_classes=self.params.num_classes)
		rscores, rbboxes = \
			tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
		# Apply NMS algorithm.
		rscores, rbboxes = \
			tfe.bboxes_nms_batch(rscores, rbboxes,
								 nms_threshold=nms_threshold,
								 keep_top_k=keep_top_k)
		if clipping_bbox is not None:
			rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
		return rscores, rbboxes


	def losses(self, logits, localisations,
			   glocalisations, gscores,
			   negative_ratio=3.,
			   use_hard_neg=False,
			   alpha=1.,
			   label_smoothing=0.,
			   scope='text_box_loss'):
		"""Define the SSD network losses.
		"""
		return text_losses(logits, localisations,
						  glocalisations, gscores,
						  match_threshold=self.params.match_threshold,
						  use_hard_neg=use_hard_neg,
						  negative_ratio=negative_ratio,
						  alpha=alpha,
						  label_smoothing=label_smoothing,
						  scope=scope)



def text_net(inputs,
			feat_layers=TextboxNet.default_params.feat_layers,
			normalizations=TextboxNet.default_params.normalizations,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			use_batch=False,
			scope='text_box_300'):
	batch_norm_params = {
	  # Decay for the moving averages.
	  'decay': 0.9997,
	  # epsilon to prevent 0s in variance.
	  'epsilon': 0.001,
	  'is_training': is_training,
	  'scale':False,
	  'fused':True,
	}
	end_points = {}
	with tf.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):
		# Original VGG-16 blocks.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		end_points['conv2'] = net # 150,150 128
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		# Block 3. # 75 75 256
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3',padding='SAME')
		# Block 4. # 38 38 512
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		end_points['conv4'] = net
		#net = slim.max_pool2d(net, [2, 2],scope='pool4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# Block 5. # 19 19 512
		#net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5',padding='SAME')

		# Additional SSD blocks.
		# Block 6: let's dilate the hell out of it!
		#net = slim.conv2d(net, 1024, [3, 3], scope='conv6')
		#net = conv2d(net, 1024, [3,3], scope='conv6',rate=1, use_batch=use_batch, batch_norm_params= batch_norm_params)
		net = conv2d(net, 1024, [3,3], scope='conv6',rate=6, use_batch=use_batch, batch_norm_params= batch_norm_params)
		end_points['conv6'] = net
		# Block 7: 1x1 conv. Because the fuck.
		#net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
		net = conv2d(net, 1024, [1, 1], scope='conv7',use_batch=use_batch, batch_norm_params= batch_norm_params)
		end_points['conv7'] = net
		# Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
		end_point = 'conv8'
		with tf.variable_scope(end_point):
			#net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
			#net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3')
			net = conv2d(net, 256, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 512, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)	
		end_points[end_point] = net
		end_point = 'conv9'
		with tf.variable_scope(end_point):
			net = conv2d(net, 128, [1, 1], scope='conv1x1', use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net
		end_point = 'conv10'
		with tf.variable_scope(end_point):
			net = conv2d(net, 128, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net
		end_point = 'global'
		with tf.variable_scope(end_point):
			net = conv2d(net, 128, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net

		print end_points
		# Prediction and localisations layers.
		logits = []
		localisations = []
		for i, layer in enumerate(feat_layers):
			with tf.variable_scope(layer + '_box'):
				p, l = text_multibox_layer(layer,
										  end_points[layer],
										  normalizations[i],
										  is_training=is_training,
										  use_batch=use_batch)
			#predictions.append(prediction_fn(p))
			logits.append(p)
			localisations.append(l)

		return localisations, logits, end_points

def conv2d(inputs, out, kernel_size, scope,stride=1,activation_fn=tf.nn.relu, 
			padding = 'SAME', use_batch=False, batch_norm_params={}, rate = 1):
	if use_batch:
		net = slim.conv2d(inputs, out, kernel_size, stride=stride ,scope=scope, normalizer_fn=slim.batch_norm, 
			  normalizer_params=batch_norm_params, activation_fn=activation_fn ,padding = padding, rate = rate)
	else:
		net = slim.conv2d(inputs, out, kernel_size, stride=stride, scope=scope, activation_fn=activation_fn,padding = padding, rate = rate)
	return net


def text_multibox_layer(layer,
					   inputs,
					   normalization=-1,
					   is_training=True,
					   use_batch=False):
	"""
	Construct a multibox layer, return a class and localization predictions.
	The  most different between textbox and ssd is the prediction shape
	where textbox has prediction score shape (38,38,2,6)
	and location has shape (38,38,2,6,4)
	besise,the kernel for fisrt 5 layers is 1*5 and padding is (0,2)
	kernel for the last layer is 1*1 and padding is 0
	"""
	batch_norm_params = {
	  # Decay for the moving averages.
	  'decay': 0.9997,
	  # epsilon to prevent 0s in variance.
	  'epsilon': 0.001,
	  'is_training': is_training,
	  'zero_debias_moving_mean':False,
	  'scale':True,
	  'fused':True,
	}
	net = inputs
	
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=False)
	
	# Number of anchors.
	num_box = 6
	num_classes = 2
	# Location.
	num_loc_pred = 2*num_box * 4

	if(layer == 'global'):
		loc_pred = conv2d(net, num_loc_pred, [1, 1], activation_fn=None, padding = 'VALID',
						   scope='conv_loc',use_batch=use_batch, batch_norm_params=batch_norm_params)
	else:
		loc_pred = conv2d(net, num_loc_pred, [1, 5], activation_fn=None, padding = 'SAME',
						   scope='conv_loc',use_batch=use_batch, batch_norm_params=batch_norm_params)

	loc_pred = custom_layers.channel_to_last(loc_pred)
	loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2, num_box,4])
	# Class prediction.
	scores_pred = 2 * num_box * num_classes

	if(layer == 'global'):
		sco_pred = conv2d(net, scores_pred, [1, 1], activation_fn=None, padding = 'VALID',
						   scope='conv_cls',use_batch=use_batch, batch_norm_params=batch_norm_params)
	else:
		sco_pred = conv2d(net, scores_pred, [1, 5], activation_fn=None, padding = 'SAME',
						   scope='conv_cls',use_batch=use_batch, batch_norm_params=batch_norm_params)

	sco_pred = custom_layers.channel_to_last(sco_pred)
	sco_pred = tf.reshape(sco_pred, sco_pred.get_shape().as_list()[:-1] + [2 ,num_box,num_classes])
	return sco_pred, loc_pred


def tensor_shape(x, rank=3):
	"""Returns the dimensions of a tensor.
	Args:
	  image: A N-D Tensor of shape.
	Returns:
	  A list of dimensions. Dimensions that are statically known are python
		integers,otherwise they are integer scalar tensors.
	"""
	if x.get_shape().is_fully_defined():
		return x.get_shape().as_list()
	else:
		static_shape = x.get_shape().with_rank(rank).as_list()
		dynamic_shape = tf.unstack(tf.shape(x), rank)
		return [s if s is not None else d
				for s, d in zip(static_shape, dynamic_shape)]




def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_regularizer=slim.l2_regularizer(weight_decay),
						weights_initializer=tf.contrib.layers.xavier_initializer(),
						biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
							padding='SAME',
							data_format=data_format):
			with slim.arg_scope([custom_layers.pad2d,
								 custom_layers.l2_normalization,
								 custom_layers.channel_to_last],
								data_format=data_format) as sc:
				return sc



# =========================================================================== #
# Text loss function.
# =========================================================================== #
def text_losses(logits, localisations,
			   glocalisations, gscores,
			   match_threshold,
			   use_hard_neg=False,
			   negative_ratio=3,
			   alpha=1.,
			   label_smoothing=0.,
			   scope=None):
	with tf.name_scope(scope, 'text_loss'):
		alllogits = []
		alllocalization = []
		for i in range(len(logits)):
			alllogits.append(tf.reshape(logits[i], [-1, 2]))
			alllocalization.append(tf.reshape(localisations[i], [-1,4]))

		alllogits = tf.concat(alllogits, 0)
		alllocalization =tf.concat(alllocalization, 0)

		pmask = gscores > match_threshold
		ipmask = tf.cast(pmask ,tf.int32)
		fpmask = tf.cast(pmask , tf.float32)
		n_pos = tf.reduce_sum(fpmask,name = 'num_of_positive')
		num = tf.ones_like(gscores, dtype = tf.float32)
		n = tf.reduce_sum(num)
		nmask = gscores <= match_threshold

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alllogits,labels=ipmask)
		#l_cross_pos = tf.losses.compute_weighted_loss(loss, fpmask)
		l_cross_pos = tf.div(tf.reduce_sum(loss * fpmask), n_pos ,name = 'l_cross_pos')
		tf.losses.add_loss(l_cross_pos)

		loss_neg = tf.where(pmask,
						   tf.cast(tf.zeros_like(ipmask),tf.float32),
						   loss)
		loss_neg_flat = tf.reshape(loss_neg, [-1])
		n_neg = tf.minimum(negative_ratio*n_pos, n)
		n_neg_i = tf.cast(n_neg, tf.int32,name = 'num_of_negative')
		val, idxes = tf.nn.top_k(loss_neg_flat, k=n_neg_i)
		minval = val[-1]
		nmask = tf.logical_and(nmask, loss_neg >= minval)

		fnmask = tf.cast(nmask, tf.float32)
		#l_cross_neg = tf.losses.compute_weighted_loss(loss, fnmask)
		l_cross_neg = tf.div(tf.reduce_sum(loss * fnmask), n_neg ,name = 'l_cross_neg')
		tf.losses.add_loss(l_cross_neg)

		weights = tf.expand_dims(fpmask, axis=-1)
		l_loc = custom_layers.abs_smooth(alllocalization - glocalisations)
		#l_loc = tf.losses.compute_weighted_loss(l_loc, weights)
		l_loc = tf.div(tf.reduce_sum(l_loc * weights), n_pos ,name = 'l_loc')
		tf.losses.add_loss(l_loc)


		with tf.name_scope('total'):
				# Add to EXTRA LOSSES TF.collection
				tf.add_to_collection('EXTRA_LOSSES', n_pos)
				tf.add_to_collection('EXTRA_LOSSES', n_neg)
				tf.add_to_collection('EXTRA_LOSSES', l_cross_pos)
				tf.add_to_collection('EXTRA_LOSSES', l_cross_neg)
				tf.add_to_collection('EXTRA_LOSSES', l_loc)

				total_loss = tf.add_n([l_loc, l_cross_pos, l_cross_neg], 'total_loss')
				tf.add_to_collection('EXTRA_LOSSES', total_loss)

	return total_loss
