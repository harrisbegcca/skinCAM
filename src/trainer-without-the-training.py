import tensorflow as tf
import os
from skincam_func_defs import *
import time

tf.reset_default_graph()
from PIL import Image

from resizeimage import resizeimage
# hyper parameters
num_classes = 9
training_iterations = 10
learning_rate = 0.00001
enable_accuracy = True
live_reply = True
total_images = 30

classes = ["Acne", "Carcinoma", "Chicken Pox", "Eczema", "Hives", "Melonoma", "Psoriasis", "Rosecea", "Warts"]

root_path = r"C:\Users\legos\OneDrive\Desktop\SkinCAM"

# following path locations
images_path = root_path + r"\input\cropped-database"
save_path = root_path + r"\output\checkpoint\CHECKPOINT.ckpt"
summary_path = root_path + r"\output\summary-log"
temp_dist_input_dir = root_path + r"\input\distributable-input"
output_tensor_file_path = root_path + r"\output\classification.log"

print("""\nHyperparameters:\n\tClasses: {}\n\tIterations: {}\n\tLearning Rate: {}\n\tEnable Accuracy: {}\n\tLive Reply: {}""".format(num_classes, training_iterations, learning_rate, enable_accuracy, live_reply))
# Declaring Placeholders
X = tf.placeholder(dtype=tf.float32, name="Placeholder_x")
Y = tf.placeholder(dtype=tf.int32, name="Placeholder_y")
# app_X = tf.placeholder(dtype=tf.float32, name="appX")

# The entire graph placed together:
convolutional, kernels_to_memory = convolutional_layer(X)
# print(convolutional, "\n", kernels_to_memory)
fully_connected, ff_weights = fully_connected_layer(convolutional, num_classes)

# here is the cost reduction drive
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_labels(Y, num_classes), logits=fully_connected, name="Loss_SOFTMAX"), name="Reduce_Mean")
optimzer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver(kernels_to_memory+ff_weights)

# starting the session
with tf.Session() as sess:

    writer = tf.summary.FileWriter(summary_path, sess.graph)
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    image_folder_dirs = os.listdir(images_path)
    image_folder_dirs.sort()
    image_folder_dirs.remove(".DS_Store") if ".DS_Store" in image_folder_dirs else None # NOTE: keyword pass doesn't work in place of 1
    image_folder_dirs = [os.path.join(images_path, image_folder) for image_folder in image_folder_dirs]

    doublList = []
    # double_list
    #image_dir_list = []
    lol_counter = None
    while True:
        print("Gimmie somthin") if lol_counter == None else print("Gimmie sem mor")
        time.sleep(2)
        if len(os.listdir(temp_dist_input_dir)) == 1:
            lol_counter = 0
            print("Om nom nom")
            test_image_path = os.path.join(temp_dist_input_dir, os.listdir(temp_dist_input_dir)[0])
            with open(test_image_path, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [100, 100])
                    cover.save(test_image_path, image.format)

            classification_image = sess.run(import_image(test_image_path))
            hypothesis = sess.run(fully_connected, feed_dict={X:classification_image})
            final_hypothesis = sess.run(tf.nn.softmax(hypothesis))

            sorted_predictions = sorted_return(final_hypothesis, classes)


            f = open(output_tensor_file_path, 'w+')
            f.write(str(sorted_predictions))
            f.close()
            # print("Written to {}. Removing image...".format(output_tensor_file_path))
            os.remove(test_image_path)
            print("Crunched them numbers real nice")