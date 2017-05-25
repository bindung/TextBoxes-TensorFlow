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

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.1       # Minimum overlap to keep a bbox after cropping.
CROP_RATIO_RANGE = (0.3, 2.)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)



def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.3, 2.0),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        labels : A Tensor inlcudes all labels
        bboxes : A Tensor inlcudes cordinates of bbox in shape [N, 4]
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=False)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes, num = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   BBOX_CROP_OVERLAP)
        return cropped_image, labels, bboxes, distort_bbox,num


def preprocess_for_train(image, labels, bboxes,
                         out_shape, data_format='NHWC',
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

        
        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        num = tf.reduce_sum(tf.cast(labels, tf.int32))
        bboxes = tf.minimum(bboxes, 1.0)
        bboxes = tf.maximum(bboxes, 0.0)
    
        # Distort image and bounding boxes.

        image, labels, bboxes, distort_bbox ,num= \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        aspect_ratio_range=CROP_RATIO_RANGE)
        
        # Resize image to output size.
        
        dst_image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        
        dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)
        #dst_image.set_shape([out_shape[0], out_shape[1], 3])
        #bbox_image = tf.image.draw_bounding_boxes(tf.expand_dims(dst_image,0), tf.expand_dims(bboxes,0))
        #tf.summary.image('image_with_box', bbox_image)

        
        dst_image = tf_image.apply_with_random_selector(
                dst_image,
                lambda x, ordering: tf_image.distort_color_2(x, ordering, False),
                num_cases=4)
        
        # Rescale to normal range
        
        image = dst_image *255
        image.set_shape([out_shape[0], out_shape[1], 3])
        image = tf_image.tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image = image/255.0
        return image, labels, bboxes,num


def preprocess_for_eval(image, labels, bboxes,
                        out_shape=EVAL_SIZE, data_format='NHWC',
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
        # Image data format.
        image = tf_image.tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image = image/255.0
        return image, labels, bboxes, bbox_img, num

def preprocess_image(image,
                     labels,
                     bboxes,
                     out_shape,
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
        return preprocess_for_train(image, labels, bboxes,
                                    out_shape=out_shape)
    else:
        return preprocess_for_eval(image, labels, bboxes,
                                   out_shape=out_shape,
                                   **kwargs)
