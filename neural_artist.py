########################################################################################
# Neural Artistic Style implementation in TensorFlow                                   # 
# By Yulun Tian, 2017                                                                  #
# Original paper: https://arxiv.org/abs/1508.06576                                     #
########################################################################################



import argparse
import tensorflow as tf
from train import train

FLAGS = None


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=1.0,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--image_width',
      type=int,
      default=300,
      help='Width of output image.'
  )
  parser.add_argument(
      '--image_height',
      type=int,
      default=300,
      help='Height of output image.'
  )
  parser.add_argument(
      '--alpha',
      type=float,
      default=1e-1,
      help='Content loss weight.'
  )
  parser.add_argument(
      '--beta',
      type=float,
      default=1e1,
      help='Style loss weight.'
  )
  parser.add_argument(
      '--gamma',
      type=float,
      default=1e0,
      help='Noise loss weight.'
  )
  parser.add_argument(
      '--content_input',
      type=str,
      default='./contents/point_reyes_in_fog.jpg',
      help='Path to input content image'
  )
  parser.add_argument(
      '--style_input',
      type=str,
      default='./styles/starry_night.jpg',
      help='Path to input style image'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='./logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='./output',
      help='Directory to put the output data.'
  )
  parser.add_argument(
      '--model',
      type=str,
      default='./vgg16/models/vgg16_weights.npz',
      help='Path to pretrained model.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default=None,
      help='Path to previous checkpoint. '
  )
  
  FLAGS, unparsed = parser.parse_known_args()

  if not tf.gfile.Exists(FLAGS.log_dir):
  	tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.output_dir):
  	tf.gfile.MakeDirs(FLAGS.output_dir)

  train(FLAGS)




