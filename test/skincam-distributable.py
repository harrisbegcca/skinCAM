# /*******************************************************
#  * Copyright (C) 2018 Ratnodeep Bandyopadhyay <ratnodeepb@gmail.com>
#  * 
#  * This file is part of skinCAM.
#  * 
#  * skincam-distributable can not be copied and/or distributed without the express
#  * permission of Ratnodeep Bandyopadhyay.
#  *******************************************************/

import tensorflow as tf
import os
import numpy as np

num_classes = 9
SAVE_PATH = os.getcwd() + "..\\intermediate\\checkpoint"
labels = ["Acne", "Carcinoma", "Chicken Pox", "Eczema", "Hives", "Melonoma", "Psoriasis", "Rosacea", "Warts"]

def import_image(parent_directory):

	image_contents = tf.read_file(parent_directory)
	image_input = tf.image.decode_jpeg(contents=image_contents, channels=3)
	modify_image_tensor = tf.image.resize_image_with_crop_or_pad(image_input, 100, 100)

	return modify_image_tensor

def directory():

	input("Directory: ")
	questions(1, labels)
	if (counter >= 2):
		print("According to the questions and our machine learning model, you have {}.\n".format(labels[1]))
	elif(counter == 0):
		print("There appears to be no disease.")
	else


#============ bias and weight functions ===============

def create_bias(shape, name):			# shape input should be an array, []
	return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name))

def create_weights(shape, name):
	return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name))

#==================== define convolutional layer =====================

def convolutional(input_image, Kernels, use_pooling=True):

	image_size = 100
	convolution = tf.reshape(input_image, [-1, image_size, image_size, 3])

	#=========== create kernels ==============

	kernel_1 = Kernels["kernel_1"]
	kernel_1_bias = Kernels["kernel_1_bias"]

	kernel_2 = Kernels["kernel_2"]
	kernel_2_bias = Kernels["kernel_2_bias"]

	#=========== first convolutional layer =============================================================================

	convolution = tf.nn.conv2d(input=convolution, filter=kernel_1, strides=[1, 1, 1, 1], padding="SAME")
	convolution = convolution + kernel_1_bias
	convolution = tf.nn.relu(convolution)

	if use_pooling:
		convolution = tf.nn.max_pool(value=convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	#========== second convolutional layer =============================================================================

	convolution = tf.nn.conv2d(input=convolution, filter=kernel_2, strides=[1, 1, 1, 1], padding="SAME")
	convolution = convolution + kernel_2_bias
	convolution = tf.nn.relu(convolution)

	if use_pooling:
		convolution = tf.nn.max_pool(value=convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

	return convolution

#===================== feed forward layer ====================

def feed_forward(input_image, feed_for_weights, use_softmax=True):

	#=============== create weight and bias layers ===============

	weight_layer_1 = feed_for_weights["weight_layer_1"]
	bias_layer_1 = feed_for_weights["bias_layer_1"]

	weight_layer_2 = feed_for_weights["weight_layer_2"]
	bias_layer_2 = feed_for_weights["bias_layer_2"]

	weight_layer_3 = feed_for_weights["weight_layer_3"]
	bias_layer_3 = feed_for_weights["bias_layer_3"]

	weight_layer_4 = feed_for_weights["weight_layer_4"]
	bias_layer_4 = feed_for_weights["bias_layer_4"]

	#=============== reshape image and store in 'x' ====================

	x = tf.reshape(input_image, [1, 625])

	#=============== link all layers ===================

	hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_1) + bias_layer_1); x = hypothesis
	hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_2) + bias_layer_2); x = hypothesis
	hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_3) + bias_layer_3); x = hypothesis
	hypothesis = tf.matmul(x, weight_layer_4) + bias_layer_4; x = hypothesis

	if use_softmax:
		hypothesis = tf.nn.softmax(x)

	return hypothesis
# ============================ create session variable ============================

sess = tf.Session()

#============================== X placeholder ==========================

X = tf.placeholder(dtype=tf.float32)

#============================== Weight Placeholders =============================

saver = tf.train.import_meta_graph(SAVE_PATH+"\\check-point.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=SAVE_PATH))

# ===================== something else ==================================

Kernel_Dict = {"kernel_1": sess.run("K1:0"), "kernel_1_bias": sess.run("KB1:0"), "kernel_2": sess.run("K2:0"), "kernel_2_bias": sess.run("KB2:0")}

Feed_Forward_Dict = {"weight_layer_1": sess.run("WL1:0"), "bias_layer_1": sess.run("BL1:0"), "weight_layer_2": sess.run("WL2:0"),
					 "bias_layer_2": sess.run("BL2:0"), "weight_layer_3": sess.run("WL3:0"), "bias_layer_3": sess.run("BL3:0"),
					 "weight_layer_4": sess.run("WL4:0"), "bias_layer_4": sess.run("BL4:0")}

#================================ pushing function declarations into variables =================

convolution_layer = convolutional(X, Kernel_Dict)
fully_connected_layer  = feed_forward(convolution_layer, Feed_Forward_Dict)

#============================== actual session =====================================

while 1:
	DIRECTORY = input(r"Directory: ")
	test_input = sess.run(import_image(DIRECTORY))
	hypo = sess.run(fully_connected_layer, feed_dict={X:test_input})

#	print(hypo)

	argmax = sess.run(tf.argmax(hypo))
#	print(argmax)
	argmax = argmax.tolist()
	highest_index = argmax.index(max(argmax))

	sec_argmax = argmax
	sec_argmax[highest_index] = 0
	second_highest_index = sec_argmax.index(max(sec_argmax))

	#print(labels[argmax.index(max(argmax))])
	#print("\n", hypo)

	questions(highest_index, labels)
#	print(counter)
	if (counter >= 2):
		print("According to the questions and our machine learning model, you have {}.\n".format(labels[highest_index]))

	elif(counter == 0):
		print("There appears to be no disease.")

	else:
		counter = 0
		questions(second_highest_index, labels)
		if (counter>= 2):
			print("According to the questions and our machine learning model, you have {}.\n".format(labels[second_highest_index]))
	directory()