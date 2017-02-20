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

  shape = (FLAGS.image_height, FLAGS.image_width)

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
    
    

    print('Begin training on ' + str((FLAGS.image_width, FLAGS.image_height)) + ' image...')

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

      sess.run(train_op)

      duration = time.time() - start_time

      # Print status to stdout.
      if step % 20 == 0 or (step + 1) == FLAGS.max_steps:
        loss_value, content_loss, style_loss, noise_loss = sess.run([loss, vgg.content_loss(content_target), vgg.style_loss(style_target), vgg.noise_loss()])
        print('Step %d: %.3f sec \n total_loss = %.3e \n content_loss = %.3e \n style_loss = %.3e \n noise_loss = %.3e' % (step, duration, loss_value, FLAGS.alpha*content_loss, FLAGS.beta*style_loss, FLAGS.gamma*noise_loss))

      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
        vgg.adjust()
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file)
        
        output_file = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '.png'
        output_file = os.path.join(FLAGS.output_dir, output_file)
        image = vgg.img.eval()
        image = np.asarray(image[0,:,:,:], dtype='uint8')
        utils.save_image(output_file, image)




