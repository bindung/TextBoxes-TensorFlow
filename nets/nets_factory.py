# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import txtbox_512
from nets import txtbox_300


slim = tf.contrib.slim

networks_map = {'text_box_300': txtbox_300.TextboxNet,
                'text_box_512': txtbox_512.TextboxNet,
               }

arg_scopes_map = {'text_box_300': txtbox_300.ssd_arg_scope,
                  'text_box_512': txtbox_512.ssd_arg_scope,
                 }
networks_obj = {'text_box_300': txtbox_300.TextboxNet,
                'text_box_512': txtbox_512.TextboxNet,
                }

def get_network(name):
    """Get a network object from a name.
    """
    # params = networks_obj[name].default_params if params is None else params
    return networks_obj[name]

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
