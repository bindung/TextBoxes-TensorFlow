
import tensorflow as tf
import numpy as np
import math
import tf_extended as tfe





# =========================================================================== #
# TensorFlow implementation of Text Boxes encoding / decoding.
# =========================================================================== #

def tf_text_bboxes_encode_layer(bboxes,
								 anchors_layer, num,box_detect,idx,
								 match_threshold=0.5,
								 prior_scaling=[0.1, 0.1, 0.2, 0.2],
								 dtype=tf.float32):
		
		"""
		Encode groundtruth labels and bounding boxes using Textbox anchors from
		one layer.

		Arguments:
			bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
			anchors_layer: Numpy array with layer anchors;
			matching_threshold: Threshold for positive match with groundtruth bboxes;
			prior_scaling: Scaling of encoded coordinates.

		Return:
			(target_localizations, target_scores): Target Tensors.
		# thisi is a binary problem, so target_score and tartget_labels are same.
		"""
		# Anchors coordinates and volume.
		

		yref, xref, href, wref = anchors_layer
		ymin = yref - href / 2.
		xmin = xref - wref / 2.
		ymax = yref + href / 2.
		xmax = xref + wref / 2. 
		vol_anchors = (xmax - xmin) * (ymax - ymin)
		
		# Initialize tensors...
		shape = (yref.shape[0], yref.shape[1], yref.shape[2], href.size)
		# all follow the shape(feat.size, feat.size, 2, 6)
		#feat_labels = tf.zeros(shape, dtype=tf.int64)
		feat_scores = tf.zeros(shape, dtype=dtype)
		feat_ymin = tf.zeros(shape, dtype=dtype)
		feat_xmin = tf.zeros(shape, dtype=dtype)
		feat_ymax = tf.ones(shape, dtype=dtype)
		feat_xmax = tf.ones(shape, dtype=dtype)

		def jaccard_with_anchors(bbox):
				"""
				Compute jaccard score between a box and the anchors.
				"""
				int_ymin = tf.maximum(ymin, bbox[0])
				int_xmin = tf.maximum(xmin, bbox[1])
				int_ymax = tf.minimum(ymax, bbox[2])
				int_xmax = tf.minimum(xmax, bbox[3])
				h = tf.maximum(int_ymax - int_ymin, 0.)
				w = tf.maximum(int_xmax - int_xmin, 0.)
				# Volumes.
				inter_vol = h * w
				union_vol = vol_anchors - inter_vol \
						+ (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
				jaccard = tf.div(inter_vol, union_vol)
				return jaccard
		


		def condition(i, feat_scores,box_detect,idx,
									feat_ymin, feat_xmin, feat_ymax, feat_xmax):
				"""Condition: check label index.
				"""
				#r = tf.less(i, tf.shape(bboxes)[0])
				r = tf.less(i, num)
				return r

		def body(i, feat_scores,box_detect,idx,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
				"""Body: update feature labels, scores and bboxes.
				Follow the original SSD paper for that purpose:
					- assign values when jaccard > 0.5;
					- only update if beat the score of other bboxes.
				"""
				# Jaccard score.

				bbox = bboxes[i]
				jaccard = jaccard_with_anchors(bbox)
				mask = tf.greater(jaccard, feat_scores)
				mask = tf.logical_and(mask, tf.greater(jaccard, match_threshold))
				imask = tf.cast(mask, tf.int64)
				fmask = tf.cast(mask, dtype)
				feat_scores = tf.where(mask, jaccard, feat_scores)

				feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
				feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
				feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
				feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

				
				max_jar = tf.reduce_max(jaccard)

				def update0(feat_scores = feat_scores,feat_ymin=feat_ymin, 
							feat_xmin=feat_xmin, feat_ymax=feat_ymax, feat_xmax=feat_xmax):
					mask = tf.equal(jaccard, max_jar)
					indices = tf.where(mask)
					indices = [indices[0]] 
					values = [1.]  
					shape = jaccard.shape
					delta = tf.SparseTensor(indices, values, shape)
					c0 = tf.zeros_like(mask, tf.float32)
					fmask = c0 + tf.sparse_tensor_to_dense(delta)

					feat_scores = tf.where(tf.cast(fmask, tf.bool), 0.51*tf.ones_like(jaccard), feat_scores)
					feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
					feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
					feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
					feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
					return feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax

				def update1(feat_scores = feat_scores,feat_ymin=feat_ymin, 
							feat_xmin=feat_xmin, feat_ymax=feat_ymax, feat_xmax=feat_xmax):
					return feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax

				def update_all(feat_scores = feat_scores,feat_ymin=feat_ymin, 
								feat_xmin=feat_xmin, feat_ymax=feat_ymax, feat_xmax=feat_xmax):
					
					feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax	= \
						tf.cond( tf.logical_and(tf.less(box_detect[i], 1),
									tf.logical_and(tf.equal(tf.reduce_sum(imask), 0), tf.greater(max_jar, 0.45))), 
							  update0, 
							  update1)
					return feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax	


				feat_scores,feat_ymin, feat_xmin, feat_ymax, feat_xmax = \
						tf.cond(tf.less(idx,0),update_all, update1)


				box_mask = tf.cast(tf.equal(tf.range(num), i), tf.int32)
				box_detect = box_detect + tf.ones([num],tf.int32) * box_mask

				return [i+1, feat_scores,box_detect,idx,
								feat_ymin, feat_xmin, feat_ymax, feat_xmax]
		# Main loop definition.

		i = 0
		[i,feat_scores,box_detect,idx,
		 feat_ymin, feat_xmin,
		 feat_ymax, feat_xmax] = tf.while_loop(condition, body,
												[i, feat_scores,box_detect,idx,
												feat_ymin, feat_xmin,
												feat_ymax, feat_xmax])



		# Transform to center / size.
		feat_cy = (feat_ymax + feat_ymin) / 2.
		feat_cx = (feat_xmax + feat_xmin) / 2.
		feat_h = feat_ymax - feat_ymin
		feat_w = feat_xmax - feat_xmin
		# Encode features.
		feat_cy = (feat_cy - yref) / href / prior_scaling[0]
		feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
		feat_h = tf.log(feat_h / href) / prior_scaling[2]
		feat_w = tf.log(feat_w / wref) / prior_scaling[3]
		# Use SSD ordering: x / y / w / h instead of ours.
		feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
		return feat_localizations, feat_scores,box_detect



def tf_text_bboxes_encode(bboxes,
						 anchors, num,
						 match_threshold=0.5,
						 prior_scaling=[0.1, 0.1, 0.2, 0.2],
						 dtype=tf.float32,
						 scope='text_bboxes_encode'):
		"""Encode groundtruth labels and bounding boxes using SSD net anchors.
		Encoding boxes for all feature layers.

		Arguments:
			bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
			anchors: List of Numpy array with layer anchors;
			matching_threshold: Threshold for positive match with groundtruth bboxes;
			prior_scaling: Scaling of encoded coordinates.

		Return:
			(target_labels, target_localizations, target_scores):
				Each element is a list of target Tensors.
		"""

		with tf.name_scope('text_bboxes_encode'):
				target_labels = []
				target_localizations = []
				target_scores = []
				box_detect = tf.zeros((num,),dtype=tf.int32)
				for i, anchors_layer in enumerate(anchors):
						with tf.name_scope('bboxes_encode_block_%i' % i):
								t_loc, t_scores,box_detect = \
										tf_text_bboxes_encode_layer(bboxes,anchors_layer, num,box_detect,i,
																	match_threshold,
																	prior_scaling, dtype)
								target_localizations.append(t_loc)
								target_scores.append(t_scores)
				return target_localizations, target_scores


## produce anchor for one layer
# each feature point has 12 default textboxes(6 boxes + 6 offsets boxes)
# aspect ratios = (1,2,3,5,7,10)
# feat_size :
		# conv4_3 ==> 38 x 38
		# fc7 ==> 19 x 19
		# conv6_2 ==> 10 x 10
		# conv7_2 ==> 5 x 5
		# conv8_2 ==> 3 x 3
		# pool6 ==> 1 x 1

def textbox_anchor_one_layer(img_shape,
														 feat_size,
														 ratios,
														 scale,
														 sizes,
														 offset = 0.5,
														 dtype=np.float32):
		# Follow the papers scheme
		# 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
		y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]]
		y = (y.astype(dtype) + offset) / feat_size[0] 
		x = (x.astype(dtype) + offset) / feat_size[1]
		y_offset = y + offset
		x_offset = x
		x_out = np.stack((x, x_offset), -1)
		y_out = np.stack((y, y_offset), -1)
		y_out = np.expand_dims(y_out, axis=-1)
		x_out = np.expand_dims(x_out, axis=-1)

		# 
		num_anchors = 6


		h = np.zeros((len(ratios), ), dtype=dtype)
		w = np.zeros((len(ratios), ), dtype=dtype)
		di = 0
		if len(sizes) > 1:
				h[0] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
				w[0] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
				di += 1
		di = 0
		for i, r in enumerate(ratios):
				h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
				w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
		return y_out, x_out, h, w



## produce anchor for all layers
def textbox_achor_all_layers(img_shape,
													 layers_shape,
													 anchor_ratios,
													 scales,
													 anchor_sizes,
													 offset=0.5,
													 dtype=np.float32):
		"""
		Compute anchor boxes for all feature layers.
		"""
		layers_anchors = []
		for i, s in enumerate(layers_shape):
				anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
																								 anchor_ratios,
																								 scales[i],
																								 anchor_sizes[i],
																								 offset=offset, dtype=dtype)
				layers_anchors.append(anchor_bboxes)
		return layers_anchors



###################
# ssd part
###################
def tf_ssd_bboxes_decode_layer(feat_localizations,
															 anchors_layer,
															 prior_scaling=[0.1, 0.1, 0.2, 0.2]):
		"""Compute the relative bounding boxes from the layer features and
		reference anchor bounding boxes.

		Arguments:
			feat_localizations: Tensor containing localization features.
			anchors: List of numpy array containing anchor boxes.

		Return:
			Tensor Nx4: ymin, xmin, ymax, xmax
		"""
		yref, xref, href, wref = anchors_layer
		# Compute center, height and width
		cx = feat_localizations[:, :, :, :, :,0] * wref * prior_scaling[0] + xref
		cy = feat_localizations[:, :, :, :, :,1] * href * prior_scaling[1] + yref
		w = wref * tf.exp(feat_localizations[:, :, :, :, :, 2] * prior_scaling[2])
		h = href * tf.exp(feat_localizations[:, :, :, :, :, 3] * prior_scaling[3])
		# Boxes coordinates.
		ymin = cy - h / 2.
		xmin = cx - w / 2.
		ymax = cy + h / 2.
		xmax = cx + w / 2.
		bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
		return bboxes


def tf_ssd_bboxes_decode(feat_localizations,
												 anchors,
												 prior_scaling=[0.1, 0.1, 0.2, 0.2],
												 scope='ssd_bboxes_decode'):
		"""Compute the relative bounding boxes from the SSD net features and
		reference anchors bounding boxes.

		Arguments:
			feat_localizations: List of Tensors containing localization features.
			anchors: List of numpy array containing anchor boxes.

		Return:
			List of Tensors Nx4: ymin, xmin, ymax, xmax
		"""
		with tf.name_scope(scope):
				bboxes = []
				for i, anchors_layer in enumerate(anchors):
						bboxes.append(
								tf_ssd_bboxes_decode_layer(feat_localizations[i],
																					 anchors_layer,
																					 prior_scaling))
				return bboxes


# =========================================================================== #
# SSD boxes selection.
# =========================================================================== #
def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
															 select_threshold=None,
															 num_classes=2,
															 ignore_class=0,
															 scope=None):
		"""Extract classes, scores and bounding boxes from features in one layer.
		Batch-compatible: inputs are supposed to have batch-type shapes.

		Args:
			predictions_layer: A SSD prediction layer;
			localizations_layer: A SSD localization layer;
			select_threshold: Classification threshold for selecting a box. All boxes
				under the threshold are set to 'zero'. If None, no threshold applied.
		Return:
			d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
				size Batches X N x 1 | 4. Each key corresponding to a class.
		"""
		select_threshold = 0.0 if select_threshold is None else select_threshold
		with tf.name_scope(scope, 'ssd_bboxes_select_layer',
											 [predictions_layer, localizations_layer]):
				# Reshape features: Batches x N x N_labels | 4
				p_shape = tfe.get_shape(predictions_layer)
				predictions_layer = tf.reshape(predictions_layer,tf.stack([p_shape[0], -1, p_shape[-1]]))
				l_shape = tfe.get_shape(localizations_layer)
				localizations_layer = tf.reshape(localizations_layer,tf.stack([l_shape[0], -1, l_shape[-1]]))

				d_scores = {}
				d_bboxes = {}
				for c in range(0, num_classes):
						if c != ignore_class:
								# Remove boxes under the threshold.
								scores = predictions_layer[:, :, c]
								fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
								scores = scores * fmask
								bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
								# Append to dictionary.
								d_scores[c] = scores
								d_bboxes[c] = bboxes

				return d_scores, d_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
												 select_threshold=None,
												 num_classes=2,
												 ignore_class=0,
												 scope=None):
		"""Extract classes, scores and bounding boxes from network output layers.
		Batch-compatible: inputs are supposed to have batch-type shapes.

		Args:
			predictions_net: List of SSD prediction layers;
			localizations_net: List of localization layers;
			select_threshold: Classification threshold for selecting a box. All boxes
				under the threshold are set to 'zero'. If None, no threshold applied.
		Return:
			d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
				size Batches X N x 1 | 4. Each key corresponding to a class.
		"""
		with tf.name_scope(scope, 'ssd_bboxes_select',
											 [predictions_net, localizations_net]):
				l_scores = []
				l_bboxes = []
				for i in range(len(predictions_net)):
						scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
																	localizations_net[i],
																	select_threshold,
																	num_classes,
																	ignore_class)
						l_scores.append(scores)
						l_bboxes.append(bboxes)
				# Concat results.
				d_scores = {}
				d_bboxes = {}
				for c in l_scores[0].keys():
						ls = [s[c] for s in l_scores]
						lb = [b[c] for b in l_bboxes]
						d_scores[c] = tf.concat(ls, axis=1)
						d_bboxes[c] = tf.concat(lb, axis=1)
				return d_scores, d_bboxes


def tf_ssd_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
																					 select_threshold=None):
		"""Extract classes, scores and bounding boxes from features in one layer.
		 Batch-compatible: inputs are supposed to have batch-type shapes.

		 Args:
			 predictions_layer: A SSD prediction layer;
			 localizations_layer: A SSD localization layer;
			select_threshold: Classification threshold for selecting a box. If None,
				select boxes whose classification score is higher than 'no class'.
		 Return:
			classes, scores, bboxes: Input Tensors.
		 """
		# Reshape features: Batches x N x N_labels | 4
		p_shape = tfe.get_shape(predictions_layer)
		predictions_layer = tf.reshape(predictions_layer,
																	 tf.stack([p_shape[0], -1, p_shape[-1]]))
		l_shape = tfe.get_shape(localizations_layer)
		localizations_layer = tf.reshape(localizations_layer,
																		 tf.stack([l_shape[0], -1, l_shape[-1]]))
		# Boxes selection: use threshold or score > no-label criteria.
		if select_threshold is None or select_threshold == 0:
				# Class prediction and scores: assign 0. to 0-class
				classes = tf.argmax(predictions_layer, axis=2)
				scores = tf.reduce_max(predictions_layer, axis=2)
				scores = scores * tf.cast(classes > 0, scores.dtype)
		else:
				sub_predictions = predictions_layer[:, :, 1:]
				classes = tf.argmax(sub_predictions, axis=2) + 1
				scores = tf.reduce_max(sub_predictions, axis=2)
				# Only keep predictions higher than threshold.
				mask = tf.greater(scores, select_threshold)
				classes = classes * tf.cast(mask, classes.dtype)
				scores = scores * tf.cast(mask, scores.dtype)
		# Assume localization layer already decoded.
		bboxes = localizations_layer
		return classes, scores, bboxes


def tf_ssd_bboxes_select_all_classes(predictions_net, localizations_net,
																		 select_threshold=None,
																		 scope=None):
		"""Extract classes, scores and bounding boxes from network output layers.
		Batch-compatible: inputs are supposed to have batch-type shapes.

		Args:
			predictions_net: List of SSD prediction layers;
			localizations_net: List of localization layers;
			select_threshold: Classification threshold for selecting a box. If None,
				select boxes whose classification score is higher than 'no class'.
		Return:
			classes, scores, bboxes: Tensors.
		"""
		with tf.name_scope(scope, 'ssd_bboxes_select',
											 [predictions_net, localizations_net]):
				l_classes = []
				l_scores = []
				l_bboxes = []
				for i in range(len(predictions_net)):
						classes, scores, bboxes = \
								tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],
																											 localizations_net[i],
																											 select_threshold)
						l_classes.append(classes)
						l_scores.append(scores)
						l_bboxes.append(bboxes)

				classes = tf.concat(l_classes, axis=1)
				scores = tf.concat(l_scores, axis=1)
				bboxes = tf.concat(l_bboxes, axis=1)
				return classes, scores, bboxes

