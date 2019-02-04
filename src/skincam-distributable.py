import tensorflow as tf
import os
from skincam_func_defs import *

distributable_input_directory = r"C:\Users\legos\OneDrive\Documents\SkinCAM\input\distributable-input"
distributable_input_image = os.path.join(distributable_input_directory, os.listdir(distributable_input_directory)[])

path_to_checkpoint = r"C:\Users\legos\OneDrive\Desktop\SkinCAM\output\checkpoint"
output_tensor_file_path = r"C:\Users\legos\OneDrive\Desktop\SkinCAM\output\classification.log"

labels = ["Acne", "Carcinoma", "Chicken Pox", "Eczema", "Hives", "Melonoma", "Psoriasis", "Roscea", "Warts"]
num_classes = 9

print("\nCreating session, and drawing metagraph...")

sess = tf.Session()
saver = tf.train.import_meta_graph(path_to_checkpoint+"/CHECKPOINT.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir= path_to_checkpoint))

print("Metagraph drawn successfully.")
# Modeling the image import
def import_image(parent_directory):

    image_contents = tf.read_file(parent_directory)
    image_input = tf.image.decode_jpeg(contents=image_contents, channels=3)
    # modify_image_tensor = tf.image.resize_image_with_crop_or_pad(image_input, 100, 100)

    return image_input

def unit_conv(n_in, num, op_max_pool=False):
    # print(num)
    if num == 0:
        kernel = sess.run("Kernel_{}:0".format(num))
    else:
        kernel = sess.run("Kernel_{}:0".format(num))

    n_out = tf.nn.conv2d(input=n_in, filter=kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv2d_{}".format(num)); n_in = n_out

    # OPTIMIZE: Need to ensure that the bias imports the proper size
    n_out = n_out.get_shape(); n_out_shape = n_out.as_list()
    n_out_shape[0]=1 if None in n_out_shape else 1

    kernel_bias = sess.run("Kernel_Bias_{}:0".format(num))
    n_out = n_in + kernel_bias; n_in = n_out
    n_out = tf.nn.relu(n_in); n_in = n_out

    if op_max_pool:
        n_out = tf.nn.max_pool(value=n_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="Pool_{}".format(num)) # ; print("Layer Pools")
    return n_out
            # op_max_pool is only used every fourth layer to simplfy computational process


# creating the convolutional layer
def convolutional_layer(input_image):
    # proto_conv = conv_unit()

    image_size = 100
    first_in = tf.reshape(input_image, [-1, image_size, image_size, 3], name="Conv_Reshape")

    total_layers = 12 # NOTE: use within in range() presents keys from 0 -> 11, but will present 1 -> 12. Look below, also need total of 4 pools
    conv_layers= []

    # NOTE: Starting bracket, resutling in construction of append
    conv_layers.append(unit_conv(first_in, 0, op_max_pool=True))

    for value in range(total_layers):
        if (value+1) % 4 == 0:
            conv_layers.append(unit_conv(conv_layers[-1], value+1, op_max_pool=True)) # ; print("entered")
        else:
            conv_layers.append(unit_conv(conv_layers[-1], value+1)) # ; print("Not entered")

    # print(conv_layers[-1])
    return conv_layers[-1]

# Defining the fully connected layer
def fully_connected_layer(input_image):
    # print(input_image.get_shape())
    x = tf.reshape(input_image, [1, -1]); print(x.get_shape())

    fc_weight = sess.run("FC_Weight:0")
    fc_bias = sess.run("FC_Bias:0")

    hypothesis = tf.matmul(x, fc_weight, name="FC_MatMul") + fc_bias

    return hypothesis

# Creating the Graph

# Declaring Placeholders
X = tf.placeholder(dtype=tf.float32, name="Placeholder_x")

print("\nConstructing convolutional layers...")
# The entire graph placed together:
convolutional = convolutional_layer(X); print("Convolutional layers constructed.\n\nConstructing fully connected...")
# print(convolutional, "\n", kernels_to_memory)
fully_connected = fully_connected_layer(convolutional); print("Fully connected layer constructed.")

print("\nImporting image...")
image_tensor = sess.run(import_image(distributable_input_image))
print("Calculating classification...")
hypothesis = sess.run(fully_connected, feed_dict={X:image_tensor})
final_hypothesis = sess.run(tf.nn.softmax(hypothesis))
sorted_predictions = sorted_return(final_hypothesis, labels)
# print(sorted_predictions)
print("Writing classification to file...")

f = open(output_tensor_file_path, 'w+')
f.write(str(sorted_predictions))
f.close()
print("Written to {}. Removing image...".format(output_tensor_file_path))
os.remove(distributable_input_image); print("image removed.")
