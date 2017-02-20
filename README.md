# Neural Artistic Style Implementation in TensorFlow

![](./presentations/illustration.png =500x)
(from "*A Neural Algorithm of Artistic Style*", Gatys et al. 2015)

By Yulun Tian [http://yuluntian.wixsite.com/yuluntian]

## Introduction
This project replicates the results in "*A Neural Algorithm of Artistic Style*", Gatys et al. 2015. A neural network is trained to create new artistic images by recombining styles and contents from different images. Total variation denoising is also integrated to ensure the output image is visually coherent.

The original paper can be accessed here: [https://arxiv.org/abs/1508.06576]

The neural network used in this project is based on a VGG16 model pre-trained on ImageNet. The model is kindly provided by Davi Frossard: [https://www.cs.toronto.edu/~frossard/post/vgg16/]

## Run this project
To run this project, first ensure you have [TensorFlow](https://www.tensorflow.org/install/) installed.

Also, download the pre-trained VGG 16 model using this [link](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz), and place the downloaded *vgg16_weights.npz* file under *vgg16/models/*.

To run a demo, run the following in your terminal:

`python neural_artist.py`

To see the full options, run:

`python neural_artist.py --help`
