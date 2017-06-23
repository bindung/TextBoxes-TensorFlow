# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Pre-processing images for textbox 
"""
from enum import Enum, IntEnum
import numpy as np

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from processing import tf_image


slim = tf.contrib.slim

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.
EVAL_SIZE = (300, 300)


def preprocess_for_train(image, labels, bboxes, height, width,
                         out_shape, data_format='NHWC',use_whiten=True,
                         scope='textbox_process_train'):
    """Preprocesses the given image for training.
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        labels : A Tensor inlcudes all labels
        bboxes : A Tensor inlcudes cordinates of bbox in shape [N, 4]
        out_shape : Image_size ,default is [300, 300]

    Returns:
        A preprocessed image.
    """

    with tf.name_scope(scope, 'textbox_process_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        bboxes = tf.minimum(bboxes, 1.0)
        bboxes = tf.maximum(bboxes, 0.0)

        image = tf_image.distorter(image)
        image,bboxes = tf_image.resize_image_bboxes_with_crop_or_pad2(image, bboxes,height[0], width[0])
    
        image, labels, bboxes = tf_image.Random_crop(image, labels, bboxes)

        image = tf_image.resize_image(image, out_shape,
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)

        image, bboxes = tf_image.random_flip_left_right(image, bboxes)
        num = tf.reduce_sum(tf.cast(labels, tf.int32))

        image.set_shape([out_shape[0], out_shape[1], 3])
        tf_image.tf_summary_image(image, bboxes)
        image = image * 255.
        image = tf_image.tf_image_whitened(image, [_R_MEAN,_G_MEAN,_B_MEAN])

        bboxes = tf.minimum(bboxes, 1.0)
        bboxes = tf.maximum(bboxes, 0.0)
        #image = tf.subtract(image, 128.)
        #image = tf.multiply(image, 2.0)
        if data_format=='NHWC':
            image = image
        else:
            image = tf.transpose(image, perm=(2, 0, 1))

        return image, labels, bboxes, num


def preprocess_for_eval(image, labels, bboxes, height, width,
                        out_shape=EVAL_SIZE, data_format='NHWC',use_whiten = True,
                        difficults=None, resize=Resize.WARP_RESIZE,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels : A Tensor inlcudes all labels
      bboxes : A Tensor inlcudes cordinates of bbox in shape [N, 4]
      out_shape : Image_size ,default is [300, 300]

    Returns:
        A preprocessed image.
    """

    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        
        
        num = 0
        if labels is not None:
            num = tf.reduce_sum(tf.cast(labels, tf.int32))
        # Add image rectangle to bboxes.
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)

        if resize == Resize.NONE:
            # No resizing...
            pass
        elif resize == Resize.CENTRAL_CROP:
            # Central cropping of the image.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.PAD_AND_RESIZE:
            # Resize image first: find the correct factor...
            shape = tf.shape(image)
            factor = tf.minimum(tf.to_double(1.0),
                                tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                           tf.to_double(out_shape[1] / shape[1])))
            resize_shape = factor * tf.to_double(shape[0:2])
            resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)

            image = tf_image.resize_image(image, resize_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
            # Pad to expected size.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.WARP_RESIZE:
            # Warp resize of the image.
            image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)

        # Split back bounding boxes.
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        # Remove difficult boxes.
        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)

        image = tf_image.tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        #image = image/255.
        #image = tf.clip_by_value(image, 0., 255.)
        #image = tf.subtract(image, 128.)
        #image = tf.multiply(image, 2.0)

        if data_format=='NHWC':
            image = image
        else:
            image = tf.transpose(image, perm=(2, 0, 1))

        return image, labels, bboxes, bbox_img, num

def preprocess_image(image,
                     labels,
                     bboxes,
                     height,
                     width,
                     out_shape,
                     data_format = 'NCHW',
                     is_training=False,
                     **kwargs):
    """Pre-process an given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      labels : A Tensor inlcudes all labels
      bboxes : A Tensor inlcudes cordinates of bbox in shape [N, 4]
      out_shape : Image_size ,default is [300, 300]

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes,height,width,data_format=data_format,
                                    out_shape=out_shape)
    else:
        return preprocess_for_eval(image, labels, bboxes,height, width,data_format=data_format,
                                   out_shape=out_shape,
                                   **kwargs)
