########################################################################################
# Neural Artistic Style implementation in TensorFlow                                   # 
# By Yulun Tian, 2017                                                                  #
# Original paper: https://arxiv.org/abs/1508.06576                                     #
########################################################################################

import numpy as np
from scipy.misc import imread, imresize, imsave


def white_noise(shape):
  noise_image = np.random.uniform(
            0, 256,
            shape).astype('uint8')
  return noise_image

def open_image(img_file, shape=None, mode='RGB'):
  img = imread(img_file, mode=mode)
  if shape is not None:
  	img = imresize(img, shape)
  return img

def save_image(img_file, img, format='JPEG'):
  imsave(name=img_file, arr=img, format=format)