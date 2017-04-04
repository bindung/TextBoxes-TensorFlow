

"""
Train scripts

"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tf_utils
import load_batch
from nets import txtbox_300

slim = tf.contrib.slim
# =========================================================================== #
# Text Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
	'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
	'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_string(
	'file_pattern', '*.tfrecord', 'tf_record pattern')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'train_dir', '/tmp/tfmodel/',
	'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
	'gpu_data', '/gpu:0',
	'Which gpu to use')
tf.app.flags.DEFINE_string(
	'gpu_train', '/gpu:0',
	'Which gpu to use')
tf.app.flags.DEFINE_integer(
	'num_readers', 4,
	'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 4,
	'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
	'log_every_n_steps', 10,
	'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
	'save_summaries_secs', 60,
	'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
	'save_interval_secs', 600,
	'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
	'gpu_memory_fraction', 0.8
	, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
	'optimizer', 'rmsprop',
	'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
	'"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
	'adadelta_rho', 0.95,
	'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
	'adagrad_initial_accumulator_value', 0.1,
	'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
	'adam_beta1', 0.9,
	'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
	'adam_beta2', 0.999,
	'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
						  'The learning rate power.')
tf.app.flags.DEFINE_float(
	'ftrl_initial_accumulator_value', 0.1,
	'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
	'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
	'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
	'momentum', 0.9,
	'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'learning_rate_decay_type',
	'exponential',
	'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
	' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
	'end_learning_rate', 0.00005,
	'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
	'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
	'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
	'num_epochs_per_decay', 1,
	'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'dataset_name', 'sythtext', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
	'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
	'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
	'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
	'labels_offset', 0,
	'An offset for the labels in the dataset. This flag is primarily used to '
	'evaluate the VGG and ResNet architectures which do not use a background '
	'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
	'model_name', 'txtbox_300', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
	'preprocessing_name', None, 'The name of the preprocessing to use. If left '
	'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
	'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
	'train_image_size', None, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', 40000,
							'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_samples', 12800,
							'Num of training set')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
	'checkpoint_path', None,
	'The path to a checkpoint from which to fine-tune.')


FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')

	tf.logging.set_verbosity(tf.logging.DEBUG)

	with tf.Graph().as_default():
		
		# initalize the net
		net = txtbox_300.TextboxNet()
		out_shape = net.params.img_shape
		anchors = net.anchors(out_shape)

		# Create global_step.
		
		global_step = slim.create_global_step()
		# create batch dataset

		b_image, b_glocalisations, b_gscores = \
		load_batch.get_batch(FLAGS.dataset_dir,
							 FLAGS.num_readers,
							 FLAGS.batch_size,
							 out_shape,
							 net,
							 anchors,
							 FLAGS.num_preprocessing_threads,
							 file_pattern = file_pattern,
							 is_training = True)
			


		with tf.device(FLAGS.gpu_train):
			#with tf.device(FLAGS.gpu_train):

			arg_scope = net.arg_scope(weight_decay=FLAGS.weight_decay)

			with slim.arg_scope(arg_scope):
				localisations, logits, end_points = \
						net.net(b_image, is_training=True)
			# Add loss function.
		with tf.device('/cpu:0'):
			total_loss = net.losses(logits, localisations,
							   b_glocalisations, b_gscores,
							   match_threshold=FLAGS.match_threshold,
							   negative_ratio=FLAGS.negative_ratio,
							   alpha=FLAGS.loss_alpha,
							   label_smoothing=FLAGS.label_smoothing)

		# Gather summaries.
		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

		for end_point in end_points:
			x = end_points[end_point]
			summaries.add(tf.summary.histogram('activations/' + end_point, x))
			summaries.add(tf.summary.scalar('sparsity/' + end_point,
											tf.nn.zero_fraction(x)))

		for loss in tf.get_collection(tf.GraphKeys.LOSSES):
			summaries.add(tf.summary.scalar(loss.op.name, loss))

		for loss in tf.get_collection('EXTRA_LOSSES'):
			summaries.add(tf.summary.scalar(loss.op.name, loss))

		for variable in slim.get_model_variables():
			summaries.add(tf.summary.histogram(variable.op.name, variable))

		with tf.device(FLAGS.gpu_train):
			learning_rate = tf_utils.configure_learning_rate(FLAGS,
															 FLAGS.num_samples,
															 global_step)
			# Configure the optimization procedure 
			optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
			summaries.add(tf.summary.scalar('learning_rate', learning_rate))

			## Training 

			train_op = slim.learning.create_train_op(total_loss, optimizer)

		# =================================================================== #
		# Kicks off the training.
		# =================================================================== #
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
		config = tf.ConfigProto(gpu_options=gpu_options,
								log_device_placement=False,
								allow_soft_placement = True)
		saver = tf.train.Saver(max_to_keep=5,
							   keep_checkpoint_every_n_hours=1.0,
							   write_version=2,
							   pad_step_number=False)
		slim.learning.train(
			train_op,
			logdir=FLAGS.train_dir,
			master='',
			is_chief=True,
			init_fn=tf_utils.get_init_fn(FLAGS),
			number_of_steps=FLAGS.max_number_of_steps,
			log_every_n_steps=FLAGS.log_every_n_steps,
			save_summaries_secs=FLAGS.save_summaries_secs,
			saver=saver,
			save_interval_secs=FLAGS.save_interval_secs,
			session_config=config,
			sync_optimizer=None)

		'''
		# for test0
		with tf.Session() as sess: 
			sess.run(tf.global_variables_initializer())
			with slim.queues.QueueRunners(sess):
				imgs = sess.run(b_image)
				loss_ = sess.run(total_loss)
				print loss_

		'''
if __name__ == '__main__':
	tf.app.run()


