

"""
Train scripts

"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tf_utils
from deployment import model_deploy
import load_batch
from nets import txtbox_300
from nets import nets_factory

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
tf.app.flags.DEFINE_integer('num_clones', 1,
							'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('shuffle_data', False,
							'Wheather shuffe the datasets')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
							'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
	'num_ps_tasks', 0,
	'The number of parameter servers. If the value is 0, then the parameters '
	'are handled locally by the worker.')
tf.app.flags.DEFINE_integer(
	'num_readers', 1,
	'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 2,
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

tf.app.flags.DEFINE_integer(
	'task', 0, 'Task id of the replica running the training.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
	'weight_decay', 0.00004, 'The weight decay on the model weights_1.')
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
tf.app.flags.DEFINE_boolean(
	'use_batch', True,
	'Wheather use batch_norm or not')
tf.app.flags.DEFINE_boolean(
	'use_hard_neg', True,
	'Wheather use use_hard_neg or not')
tf.app.flags.DEFINE_boolean(
	'use_whiten', True,
	'Wheather use whiten or not,genally you can choose whiten or batchnorm tech.')

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
	'model_name', 'text_box_300', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
	'data_format', 'NCHW', 'data format.')
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
tf.app.flags.DEFINE_string(
	'checkpoint_model_scope', None,
	'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
	'checkpoint_exclude_scopes', None,
	'Comma-separated list of scopes of variables to exclude when restoring '
	'from a checkpoint.')
tf.app.flags.DEFINE_string(
	'trainable_scopes', None,
	'Comma-separated list of scopes to filter the set of variables to train.'
	'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
	'ignore_missing_vars', False,
	'When restoring a checkpoint would ignore missing variables.')


FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')

	tf.logging.set_verbosity(tf.logging.DEBUG)

	with tf.Graph().as_default():
		######################
		# Config model_deploy#
		######################
		deploy_config = model_deploy.DeploymentConfig(
			num_clones=FLAGS.num_clones,
			clone_on_cpu=FLAGS.clone_on_cpu,
			replica_id=FLAGS.task,
			num_replicas=FLAGS.worker_replicas,
			num_ps_tasks=FLAGS.num_ps_tasks)

		# Create global_step
		with tf.device(deploy_config.variables_device()):
			global_step = slim.create_global_step()


		network_fn = nets_factory.get_network(FLAGS.model_name)
		params = network_fn.default_params
		params = params._replace( match_threshold=FLAGS.match_threshold)
		# initalize the net
		net = network_fn(params)
		out_shape = net.params.img_shape
		anchors = net.anchors(out_shape)

		# create batch dataset
		with tf.device(deploy_config.inputs_device()):
			b_image, b_glocalisations, b_gscores = \
			load_batch.get_batch(FLAGS.dataset_dir,
								 FLAGS.num_readers,
								 FLAGS.batch_size,
								 out_shape,
								 net,
								 anchors,
								 FLAGS,
								 file_pattern = FLAGS.file_pattern,
								 is_training = True,
								 shuffe = FLAGS.shuffle_data)
				
			batch_queue = slim.prefetch_queue.prefetch_queue(
				tf_utils.reshape_list([b_image, b_glocalisations, b_gscores]),
				capacity=10 * deploy_config.num_clones)


		# =================================================================== #
		# Define the model running on every GPU.
		# =================================================================== #
		def clone_fn(batch_queue):
			
			#Allows data parallelism by creating multiple
			#clones of network_fn. 
			
			# Dequeue batch.
			batch_shape = [1] + [len(anchors)] * 2
			b_image, b_glocalisations, b_gscores = \
				tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)

			if FLAGS.data_format=='NHWC':
				b_image = b_image
			else:
				b_image = tf.transpose(b_image, perm=(0, 3, 1, 2))
			# Construct SSD network.
			arg_scope = net.arg_scope(weight_decay=FLAGS.weight_decay,data_format=FLAGS.data_format)
			with slim.arg_scope(arg_scope):
				localisations, logits, end_points = \
					net.net(b_image, is_training=True, use_batch=FLAGS.use_batch)
			# Add loss function.
			net.losses(logits, localisations,
							   b_glocalisations, b_gscores,
							   negative_ratio=FLAGS.negative_ratio,
							   use_hard_neg=FLAGS.use_hard_neg,
							   alpha=FLAGS.loss_alpha,
							   label_smoothing=FLAGS.label_smoothing)
			return end_points

		

		clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
		first_clone_scope = deploy_config.clone_scope(0)

		# Gather summaries.
		summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES,first_clone_scope))
		# Gather update_ops from the first clone. These contain, for example,
		# the updates for the batch_norm variables created by network_fn.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

		
		end_points = clones[0].outputs
		for end_point in end_points:
			x = end_points[end_point]
			summaries.add(tf.summary.histogram('activations/' + end_point, x))
			#summaries.add(tf.summary.scalar('sparsity/' + end_point,
			#								tf.nn.zero_fraction(x)))
		

		#for loss in tf.get_collection(tf.GraphKeys.LOSSES):
		#	summaries.add(tf.summary.scalar(loss.op.name, loss))
		# Add summaries for losses.
		#for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
		#  summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

		for loss in tf.get_collection('EXTRA_LOSSES',first_clone_scope):
			summaries.add(tf.summary.scalar(loss.op.name, loss))

		
		for variable in slim.get_model_variables():
			summaries.add(tf.summary.histogram(variable.op.name, variable))
		
		#################################
		# Configure the moving averages #
		#################################
		if FLAGS.moving_average_decay:
			moving_average_variables = slim.get_model_variables()
			variable_averages = tf.train.ExponentialMovingAverage(
							  FLAGS.moving_average_decay, global_step)
		else:
			moving_average_variables, variable_averages = None, None

		#########################################
		# Configure the optimization procedure. #
		#########################################
		with tf.device(deploy_config.optimizer_device()):
			learning_rate = tf_utils.configure_learning_rate(FLAGS,
															 FLAGS.num_samples,
															 global_step)
			optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
			summaries.add(tf.summary.scalar('learning_rate', learning_rate))

		if FLAGS.moving_average_decay:
			# Update ops executed locally by trainer.
			update_ops.append(variable_averages.apply(moving_average_variables))

		# Variables to train.
		variables_to_train = tf_utils.get_variables_to_train(FLAGS)

		#  and returns a train_tensor and summary_op
		total_loss, clones_gradients = model_deploy.optimize_clones(
			clones,
			optimizer,
			var_list=variables_to_train)
		# Add total_loss to summary.
		#summaries.add(tf.summary.scalar('total_loss', total_loss))

		# Create gradient updates.
		grad_updates = optimizer.apply_gradients(clones_gradients,
												 global_step=global_step)
		update_ops.append(grad_updates)

		update_op = tf.group(*update_ops)
		train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
														  name='train_op')

		# Add the summaries from the first clone. These contain the summaries
		# created by model_fn and either optimize_clones() or _gather_clone_loss().
		#summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
		#								   first_clone_scope))

		# Merge all summaries together.
		summary_op = tf.summary.merge(list(summaries), name='summary_op')

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
			train_tensor,
			logdir=FLAGS.train_dir,
			master='',
			is_chief=True,
			init_fn=tf_utils.get_init_fn(FLAGS),
			summary_op=summary_op,
			number_of_steps=FLAGS.max_number_of_steps,
			log_every_n_steps=FLAGS.log_every_n_steps,
			save_summaries_secs=FLAGS.save_summaries_secs,
			saver=saver,
			save_interval_secs=FLAGS.save_interval_secs,
			session_config=config,
			sync_optimizer=None)

if __name__ == '__main__':
	tf.app.run()


