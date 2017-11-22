import math
import time

import numpy as np
import tensorflow as tf
import tf_extended as tfe
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import load_batch
from nets import nets_factory
slim = tf.contrib.slim

# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
				0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'select_threshold', 0.1, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
	'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
	'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
	'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
	'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
	'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
	'eval_image_size', None, 'Eval image size.')
tf.app.flags.DEFINE_boolean(
	'remove_difficult', True, 'Remove difficult objects from evaluation.')
tf.app.flags.DEFINE_integer(
	'num_samples', 229, 'number of dataset size')
tf.app.flags.DEFINE_string(
    'model_name', 'text_box_300', 'The name of the architecture to evaluate.')


# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
	'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
	'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
	'max_num_batches', None,
	'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string(
	'master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
	'checkpoint_path', './checkpoints/model.ckpt-33763',
	'The directory where the model was written to or an absolute path to a '
	'checkpoint file.')
tf.app.flags.DEFINE_string(
	'eval_dir', './data/eval/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_integer(
	'num_readers', 4,
	'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 4,
	'The number of threads used to create the batches.')
tf.app.flags.DEFINE_string(
	'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_float(
	'gpu_memory_fraction', 0.08, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_string(
	'gpu_eval', '/cpu:0',
	'Which gpu to use')
tf.app.flags.DEFINE_boolean(
	'wait_for_checkpoints', False, 'Wait for new checkpoints in the eval loop.')
tf.app.flags.DEFINE_integer('shuffle_data', False,
							'Wheather shuffe the datasets')
tf.app.flags.DEFINE_boolean(
	'use_batch', True,
	'Wheather use batch_norm or not')
tf.app.flags.DEFINE_boolean(
	'use_whiten', True,
	'Wheather use whiten or not,genally you can choose whiten or batchnorm tech.')


FLAGS = tf.app.flags.FLAGS


def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')

	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.Graph().as_default():
		tf_global_step = slim.get_or_create_global_step()

		# initalize the net
		network_fn = nets_factory.get_network(FLAGS.model_name)
		net = network_fn()
		out_shape = net.params.img_shape
		out_shape = (300,300)
		anchors = net.anchors(out_shape)
		# =================================================================== #
		# Create a dataset provider and batches.
		# =================================================================== #
		with tf.device('/cpu:0'):
			b_image, glabels, b_gbboxes, g_bbox_img, b_glocalisations, b_gscores =\
							load_batch.get_batch(FLAGS.dataset_dir,
										 FLAGS.num_readers,
										 FLAGS.batch_size,
										 out_shape,
										 net,
										 anchors,
										 FLAGS,
										 file_pattern =  '*.tfrecord',
										 is_training = False,
										 shuffe = FLAGS.shuffle_data)
		b_gdifficults = tf.zeros(tf.shape(glabels), dtype=tf.int64)
		dict_metrics = {}
		arg_scope = net.arg_scope(data_format=DATA_FORMAT)
		with slim.arg_scope(arg_scope):
			localisations, logits, end_points  = \
				net.net(b_image, is_training=False, use_batch=FLAGS.use_batch)
		# Add losses functions.
		#total_loss = net.losses(logits, localisations,
		#					  b_glocalisations, b_gscores)
		predictions = []
		for i in range(len(logits)):
			predictions.append(slim.softmax(logits[i]))
		
		# Performing post-processing on CPU: loop-intensive, usually more efficient.
		with tf.device('/device:CPU:0'):
			# Detected objects from SSD output.
			localisations = net.bboxes_decode(localisations, anchors)
			rscores, rbboxes = \
				net.detected_bboxes(predictions, localisations,
										select_threshold=FLAGS.select_threshold,
										nms_threshold=FLAGS.nms_threshold,
										clipping_bbox=None,
										top_k=FLAGS.select_top_k,
										keep_top_k=FLAGS.keep_top_k)
			# Compute TP and FP statistics.
			num_gbboxes, tp, fp, rscores = \
				tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
										  glabels, b_gbboxes, b_gdifficults,
										  matching_threshold=FLAGS.matching_threshold)

		# Variables to restore: moving avg. or normal weights.
		if FLAGS.moving_average_decay:
			variable_averages = tf.train.ExponentialMovingAverage(
				FLAGS.moving_average_decay, tf_global_step)
			variables_to_restore = variable_averages.variables_to_restore(
				slim.get_model_variables())
			variables_to_restore[tf_global_step.op.name] = tf_global_step
		else:
			variables_to_restore = slim.get_variables_to_restore()

		# =================================================================== #
		# Evaluation metrics.
		# =================================================================== #
		with tf.device(FLAGS.gpu_eval):
			dict_metrics = {}
			# Extra losses as well.
			for loss in tf.get_collection('EXTRA_LOSSES'):
				dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)

			# Add metrics to summaries and Print on screen.
			for name, metric in dict_metrics.items():
				# summary_name = 'eval/%s' % name
				summary_name = name
				op = tf.summary.scalar(summary_name, metric[0], collections=[])
				# op = tf.Print(op, [metric[0]], summary_name)
				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

			# FP and TP metrics.
			tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
			for c in tp_fp_metric[0].keys():
				dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
												tp_fp_metric[1][c])

			# Add to summaries precision/recall values.
			icdar2013 = {}
			for c in tp_fp_metric[0].keys():
				# Precison and recall values.
				prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

				op = tf.summary.scalar('precision', tf.reduce_mean(prec), collections=[])
				# op = tf.Print(op, [v], summary_name)
				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

				op = tf.summary.scalar('recall', tf.reduce_mean(rec), collections=[])
				# op = tf.Print(op, [v], summary_name)
				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

				# Average precision VOC07.
				v = tfe.average_precision_voc12(prec, rec)				
				#v = (prec + rec)/2.
				summary_name = 'ICDAR13/%s' % c
				op = tf.summary.scalar(summary_name, v, collections=[])
				# op = tf.Print(op, [v], summary_name)
				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
				icdar2013[c] = v


			# Mean average precision VOC07.
			summary_name = 'ICDAR13/mAP'
			mAP = tf.add_n(list(icdar2013.values())) / len(icdar2013)
			op = tf.summary.scalar(summary_name, mAP, collections=[])
			op = tf.Print(op, [mAP], summary_name)
			tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


		# Split into values and updates ops.
		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

		# =================================================================== #
		# Evaluation loop.
		# =================================================================== #
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
		config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
		# config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

		# Number of batches...
		if FLAGS.max_num_batches:
			num_batches = FLAGS.max_num_batches
		else:
			num_batches = math.ceil(FLAGS.num_samples / float(FLAGS.batch_size))

		if not FLAGS.wait_for_checkpoints:
			if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
				checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
			else:
				checkpoint_path = FLAGS.checkpoint_path
			tf.logging.info('Evaluating %s' % checkpoint_path)

			# Standard evaluation loop.
			start = time.time()
			slim.evaluation.evaluate_once(
				master=FLAGS.master,
				checkpoint_path=checkpoint_path,
				logdir=FLAGS.eval_dir,
				num_evals=num_batches,
				eval_op=list(names_to_updates.values()),
				variables_to_restore=variables_to_restore,
				session_config=config)
			# Log time spent.
			elapsed = time.time()
			elapsed = elapsed - start
			print('Time spent : %.3f seconds.' % elapsed)
			print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))

		else:
			checkpoint_path = FLAGS.checkpoint_path
			tf.logging.info('Evaluating %s' % checkpoint_path)

			# Waiting loop.
			slim.evaluation.evaluation_loop(
				master=FLAGS.master,
				checkpoint_dir=checkpoint_path,
				logdir=FLAGS.eval_dir,
				num_evals=num_batches,
				eval_op=list(names_to_updates.values()),
				variables_to_restore=variables_to_restore,
				eval_interval_secs=60,
				max_number_of_evaluations=np.inf,
				session_config=config,
				timeout=None)


if __name__ == '__main__':
	tf.app.run()
