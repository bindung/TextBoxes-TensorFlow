"""Test the tf_image functions
function list:
    Random_Brightness
"""

import tensorflow as tf
import numpy as np
from processing import tf_image
slim  = tf.contrib.slim

class test_image_augmentation(tf.test.TestCase):

    def test_random_brightness(self):
        image = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
        #image_tensor = tf.image.encode_jpeg(tf.constant(image))
        image_tensor = tf.to_float(tf.constant(image))/255.
        with self.test_session():
            image_ = image_tensor.eval()
            image_aug = tf_image.Random_Brightness(image_tensor).eval()
            print image_ == image_aug


if __name__ == '__main__':
    tf.test.main()
