import tensorflow as tf
# import numpy as np
# import decimal

# Creating labels
def y_labels(index, classes):
    thing = tf.one_hot(index, classes, name="Label_Maker")
    # print(thing)
    return thing

# imports and contextualizes images
def import_image(parent_directory):

    image_contents = tf.read_file(parent_directory)
    image_input = tf.image.decode_jpeg(contents=image_contents, channels=3)
    #modify_image_tensor = tf.image.resize_image_with_crop_or_pad(image_input, 100, 100)

    return image_input

# Defining the functions for creating weights and biases
def create_bias(shape, name):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name)) # NOTE: shape should be an array

def create_weights(shape, name):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.05, name=name))

def sorted_return(result_tensor, label_list):      # NOTE: needs to pump in an ndarray
    result_tensor.tolist(); print(result_tensor)
    temp_list = []
    temp_list_2 = []
    for iterator in range(9):
        temp_list.append(result_tensor[0][iterator])
    for i in temp_list:
        if i<0.00001:
            temp_list_2.append(0)
        else:
            temp_list_2.append(i)
    # temp_list_2 = [0 if i in temp_list else i]
    # print(temp_list); print(type(temp_list))
    zipped_result = list(zip(temp_list_2, label_list))
    zipped_result.sort(reverse=True)
    return zipped_result

# creating the convolutional layer
def convolutional_layer(input_image):

    image_size = 100
    first_in = tf.reshape(input_image, [-1, image_size, image_size, 3], name="Conv_Reshape")

    kernel_array = []
    # unit_conv simplifies establishing the convolutional layers
    def unit_conv(n_in, num, op_max_pool=False):
        # print(num)
        if num == 0:
            kernel = create_weights(shape=[10, 10, 3, 10], name="Kernel_{}".format(num))
        else:
            kernel = create_weights(shape=[10, 10, 10, 10], name="Kernel_{}".format(num))
        kernel_array.append(kernel)

        n_out = tf.nn.conv2d(input=n_in, filter=kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv2d_{}".format(num)); n_in = n_out
        #
        # if num == 0:
        #     n_out = tf.nn.dropout(n_in, 0.95); n_in = n_out

        # OPTIMIZE: Need to ensure that the bias imports the proper size
        n_out = n_out.get_shape(); n_out_shape = n_out.as_list()
        n_out_shape[0]=1 if None in n_out_shape else 1

        kernel_bias = create_weights(shape=n_out_shape, name="Kernel_Bias_{}".format(num)); kernel_array.append(kernel_bias)
        n_out = n_in + kernel_bias; n_in = n_out
        n_out = tf.nn.relu(n_in, name="relu_{}".format(num)); n_in = n_out
        # op_max_pool is only used every fourth layer to simplfy computational process
        if op_max_pool:
            n_out = tf.nn.max_pool(value=n_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="Pool_{}".format(num)) # ; print("Layer Pools")
        return n_out

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

    return conv_layers[-1], kernel_array

# Defining the fully connected layer
def fully_connected_layer(input_image, classes):
    # print(input_image.get_shape())
    x = tf.reshape(input_image, [1, -1]) #; print(x.get_shape())

    fc_weight = create_weights(shape=[490, 9], name="FC_Weight")
    fc_bias = create_bias(shape=(1, classes), name="FC_Bias")

    fully_connected_weights=[fc_bias, fc_weight]
    hypothesis = tf.matmul(x, fc_weight, name="FC_MatMul") + fc_bias; input = hypothesis

    n_out = tf.nn.dropout(input, 0.95)

    # tf.summary.histogram("Weight 1", weight_layer_1)
    # tf.summary.histogram("Bias 1", bias_layer_1)

    return hypothesis, fully_connected_weights
