# /*******************************************************
#  * Copyright (C) 2018 Ratnodeep Bandyopadhyay <ratnodeepb@gmail.com>
#  * 
#  * This file is part of skinCAM.
#  * 
#  * skincam can not be copied and/or distributed without the express
#  * permission of Ratnodeep Bandyopadhyay.
#  *******************************************************/

import tensorflow as tf
from tqdm import tqdm
import os
# ============ hyper parameters =============

num_classes = 9
training_iterations = 1
learning_rate = 0.1
PATH = os.getcwd() + "..\\input\\cropped-database"
SAVE_PATH = os.getcwd() + "..\\intermediate\\checkpoint\\check-point.ckpt"
print("\nHyperparameters:\n\tClasses: {}\n\tIterations: {}\n\tLearning Rate: {}\n".format(num_classes, training_iterations, learning_rate))

# ====================== label creation ==================

def y_labels(index, classes):
    return tf.one_hot(index, classes)

# ============ image import ================

def import_image(parent_directory):

    image_contents = tf.read_file(parent_directory)
    image_input = tf.image.decode_jpeg(contents=image_contents, channels=3)
    #modify_image_tensor = tf.image.resize_image_with_crop_or_pad(image_input, 100, 100)

    return image_input

# ============ bias and weight functions ===============

def create_bias(shape, name):			# shape input should be an array, []
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name))

def create_weights(shape, name):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name))

# ==================== define convolutional layer =====================

def convolutional(input_image, Kernels, use_pooling=True):

    image_size = 100
    convolution = tf.reshape(input_image, [-1, image_size, image_size, 3])

    # =========== create kernels ==============

    kernel_1 = Kernels["kernel_1"]
    kernel_1_bias = Kernels["kernel_1_bias"]

    kernel_2 = Kernels["kernel_2"]
    kernel_2_bias = Kernels["kernel_2_bias"]

    # =========== first convolutional layer =============================================================================

    convolution = tf.nn.conv2d(input=convolution, filter=kernel_1, strides=[1, 1, 1, 1], padding="SAME")
    convolution = convolution + kernel_1_bias
    convolution = tf.nn.relu(convolution)

    if use_pooling:
        convolution = tf.nn.max_pool(value=convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #tf.summary.image("convolutional 1", convolution, 3)

    # ========== second convolutional layer =============================================================================

    convolution = tf.nn.conv2d(input=convolution, filter=kernel_2, strides=[1, 1, 1, 1], padding="SAME")
    convolution = convolution + kernel_2_bias
    convolution = tf.nn.relu(convolution)

    if use_pooling:
        convolution = tf.nn.max_pool(value=convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return convolution

# ===================== feed forward layer ====================
with tf.name_scope("ff"):
    def feed_forward(input_image, feed_for_weights):

        # =============== create weight and bias layers ===============

        weight_layer_1 = feed_for_weights["weight_layer_1"]
        bias_layer_1 = feed_for_weights["bias_layer_1"]

        weight_layer_2 = feed_for_weights["weight_layer_2"]
        bias_layer_2 = feed_for_weights["bias_layer_2"]

        weight_layer_3 = feed_for_weights["weight_layer_3"]
        bias_layer_3 = feed_for_weights["bias_layer_3"]

        weight_layer_4 = feed_for_weights["weight_layer_4"]
        bias_layer_4 = feed_for_weights["bias_layer_4"]

        # =============== reshape image and store in 'x' ====================

        x = tf.reshape(input_image, [1, 625])

        # =============== link all layers ===================

        hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_1) + bias_layer_1); x = hypothesis
        hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_2) + bias_layer_2); x = hypothesis
        hypothesis = tf.nn.relu(tf.matmul(x, weight_layer_3) + bias_layer_3); x = hypothesis
        hypothesis = tf.matmul(x, weight_layer_4) + bias_layer_4

        tf.summary.histogram("Weight 1", weight_layer_1)
        tf.summary.histogram("Bias 1", bias_layer_1)

        return hypothesis

# =================== Declaring placeholders ====================

with tf.name_scope("Placeholders"):
    X = tf.placeholder(dtype=tf.float32)
    Y = tf.placeholder(dtype=tf.int32)

# ===========================Variable Declarations======================
if input("Use stored session image?: [Y/n] ").upper() == "Y":
    SESSION = tf.Session()
    saver = tf.train.import_meta_graph(os.getcwd()+"\\checkpoint\\check-point.ckpt.meta")
    saver.restore(SESSION, tf.train.latest_checkpoint(checkpoint_dir=os.getcwd() + "\\checkpoint"))

    Kernel_Dict = {"kernel_1": tf.Variable(SESSION.run("K1:0")), "kernel_1_bias": tf.Variable(SESSION.run("KB1:0")), "kernel_2": tf.Variable(SESSION.run("K2:0")),
                   "kernel_2_bias": tf.Variable(SESSION.run("KB2:0"))}

    Feed_Forward_Dict = {"weight_layer_1": tf.Variable(SESSION.run("WL1:0")), "bias_layer_1": tf.Variable(SESSION.run("BL1:0")), "weight_layer_2": tf.Variable(SESSION.run("WL2:0")),
                         "bias_layer_2": tf.Variable(SESSION.run("BL2:0")), "weight_layer_3": tf.Variable(SESSION.run("WL3:0")), "bias_layer_3": tf.Variable(SESSION.run("BL3:0")),
                         "weight_layer_4": tf.Variable(SESSION.run("WL4:0")), "bias_layer_4": tf.Variable(SESSION.run("BL4:0"))}
else:
    Kernel_Dict = {"kernel_1": create_weights(shape=[25, 25, 3, 1], name="K1"), "kernel_1_bias": create_bias(shape=[1, 100, 100, 1], name="KB1"),
                   "kernel_2": create_weights(shape=[25, 25, 1, 1], name="K2"), "kernel_2_bias": create_bias(shape=[1, 50, 50, 1], name="KB2")}

    Feed_Forward_Dict = {"weight_layer_1": create_weights(shape=(625, 500), name="WL1"), "bias_layer_1": create_bias(shape=(1, 1), name="BL1"),
                         "weight_layer_2": create_weights(shape=(500, 350), name="WL2"), "bias_layer_2": create_bias(shape=(1, 1), name="BL2"),
                         "weight_layer_3": create_weights(shape=(350, 100), name="WL3"), "bias_layer_3": create_bias(shape=(1, 1), name="BL3"),
                         "weight_layer_4": create_weights(shape=(100, num_classes), name="WL4"), "bias_layer_4": create_bias(shape=(1, 1), name="BL4")}

# ============================ Assigning Everything ============================

convolution_layer = convolutional(X, Kernel_Dict)
fully_connected_layer  = feed_forward(convolution_layer, Feed_Forward_Dict)
with tf.name_scope("cost"):
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels(index=Y, classes=num_classes), logits=fully_connected_layer))
    tf.summary.scalar("cost", cost)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

SAVER = tf.train.Saver(var_list=[Kernel_Dict["kernel_1"], Kernel_Dict["kernel_2"],Kernel_Dict["kernel_1_bias"], Kernel_Dict["kernel_2_bias"],
                                 Feed_Forward_Dict["weight_layer_1"], Feed_Forward_Dict["weight_layer_2"], Feed_Forward_Dict["weight_layer_3"],
                                 Feed_Forward_Dict["weight_layer_4"], Feed_Forward_Dict["bias_layer_1"], Feed_Forward_Dict["bias_layer_2"],
                                 Feed_Forward_Dict["bias_layer_3"], Feed_Forward_Dict["bias_layer_4"]], max_to_keep=1)

# ============================ Session =================================
with tf.Session() as sess:

    writer = tf.summary.FileWriter(os.getcwd()+"..\\output\\summary", sess.graph)
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # BABAI: In this block, I'm stuck with having to parse through every folder in the database folder, and it's really annoying.

    image_folder_dirs = os.listdir(PATH)
    image_folder_dirs.remove(".DS_Store") if ".DS_Store" in image_folder_dirs else 1
    image_folder_dirs = [os.path.join(PATH, image_folder) for image_folder in image_folder_dirs]

    doublList = []
    #image_dir_list = []

    for i in range(len(image_folder_dirs)): #for folder in image_folder_dirs: ? - NO because I use 'i' in feed_dict={Y:i}

        os.chdir(image_folder_dirs[i])          #changes into the directory
        image_names = os.listdir()          #lists elements in the directory
        image_names.remove(".DS_Store") if ".DS_Store" in image_names else 1

        image_dir_list = [os.path.join(os.getcwd(), file_name) for file_name in image_names]
#        print("image_dir_list:\n", image_dir_list, end="\n")
        # add this new list to our two dimentional list
        doublList.append(image_dir_list)
#        print("doublList:\n", doublList, end="\n")
        # delete the temp list
#        del image_dir_list[:]
        os.chdir("..")

#    print("doublList:\n", doublList, end="\n")

    summary = 0

    for iteration in tqdm(range(training_iterations), desc="COMPLETION", ncols=80):             #declaring the number of iterations
        i = 0
        for myList in doublList:
#            print("myList:\n", myList, end="\n")
            for file in myList:
#                print("file:\n", file, end="\n")
                image_tensor = sess.run(import_image(file))
                print(image_tensor, "END OF TENSOR:", file)
                tf.summary.image(name="image", tensor=image_tensor)
                summary, _ = sess.run([merged, train_step], feed_dict={X:image_tensor, Y:i})

            i = i + 1
        SAVER.save(sess, SAVE_PATH)
        writer.add_summary(summary)

    print("\n\nTraining completed.\nRunning test prediction.\n")

    while 1:
        DIRECTORY = input(r"Directory: ")
        test_input = sess.run(import_image(DIRECTORY))
        hypo = sess.run(fully_connected_layer, feed_dict={X:test_input})

        print(hypo)

# to view tensorboard:   python -m tensorboard.main --logdir=
