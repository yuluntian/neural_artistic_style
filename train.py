########################################################################################
# Neural Artistic Style implementation in TensorFlow                                   # 
# By Yulun Tian, 2017                                                                  #
# Original paper: https://arxiv.org/abs/1508.06576                                     #
########################################################################################

import tensorflow as tf
import numpy as np
from vgg16.vgg16 import vgg16
import utils
import time
import os.path


def train(FLAGS):

  shape = (FLAGS.image_width, FLAGS.image_height)

  content_img, style_img = utils.open_image(FLAGS.content_input, shape=shape), utils.open_image(FLAGS.style_input, shape=shape)

  with tf.Session() as sess:

    print('Calculating target content...')

    vgg = vgg16(img=content_img, train=False) 

    sess.run(tf.global_variables_initializer())

    vgg.load_weights(weight_file=FLAGS.model, sess=sess)
    
    content_target = sess.run(vgg.content_response())

    

    print('Calculating target style...')

    vgg = vgg16(img=style_img, train=False) 

    sess.run(tf.global_variables_initializer())

    vgg.load_weights(weight_file=FLAGS.model, sess=sess)
    
    style_target = sess.run(vgg.style_response())
    
    

    print('Begin training on ' + str(shape) + ' image...')

    vgg = vgg16(img=utils.white_noise(shape=shape+(3,)), train=True)

    loss = FLAGS.alpha * vgg.content_loss(content_target) + FLAGS.beta * vgg.style_loss(style_target) + FLAGS.gamma * vgg.noise_loss()

    train_op = vgg.training(loss, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver()

    if FLAGS.checkpoint is not None:
      saver.restore(sess, FLAGS.checkpoint)
      print("Progress restored.")
    else:
      sess.run(tf.global_variables_initializer())
      vgg.load_weights(weight_file=FLAGS.model, sess=sess)

    for step in range(FLAGS.max_steps):
      start_time = time.time()

      _, loss_value= sess.run([train_op, loss])
      
      duration = time.time() - start_time

      # Print status to stdout.
      print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
        vgg.adjust()
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file)
        
        output_file = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '.png'
        output_file = os.path.join(FLAGS.output_dir, output_file)
        image = vgg.img.eval()
        image = np.asarray(image[0,:,:,:], dtype='uint8')
        utils.save_image(output_file, image)




