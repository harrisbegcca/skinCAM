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

exit() if input("\nProceed? [Y/n] ").lower() == "n" else 1

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

    for i in range(len(image_folder_dirs)): #for folder in image_folder_dirs: ? - NO because I use 'i' in feed_dict={Y:i}

        os.chdir(image_folder_dirs[i])          #changes into the directory
        image_names = os.listdir()          #lists elements in the directory
        image_names.remove(".DS_Store") if ".DS_Store" in image_names else None # NOTE: keyword pass doesn't work in place of 1

        image_dir_list = [os.path.join(os.getcwd(), file_name) for file_name in image_names]
        # print("image_dir_list:\n", image_dir_list, end="\n")
        # add this new list to our two dimentional list
        doublList.append(image_dir_list)
        # print("doublList:\n", doublList, end="\n")
        # delete the temp list
        # del image_dir_list[:]
        os.chdir("..")

    # print("doublList:\n", doublList, end="\n")

    summary, acc_count = 0, 0 # NOTE: Initializing the summary value so as to create the summary value appropriately

    for epoch in range(training_iterations):
        i = 0
        mem_epoch = epoch + 1
        for myList in doublList:
            # print("myList:\n", myList, end="\n")
            for file in myList:
                # print("file:\n", file, end="\n")
                image_tensor = sess.run(import_image(file))
                # print(image_tensor, "END OF TENSOR:", file)
                if enable_accuracy:
                    _, acc_hypo= sess.run([optimzer, fully_connected], feed_dict={X:image_tensor, Y:i}) # NOTE: make into list?

                    # print(sess.run(tf.argmax(acc_hypo, axis=1)), "ACCURACY")#; print(acc_hypo)
                    # print(sess.run(tf.argmax(y_labels(i, num_classes), axis=0)), "LABLES")#; print(sess.run(y_labels(i, num_classes)))

                    # print(sess.run(tf.argmax(acc_hypo, axis=1))[0] == sess.run(tf.argmax(y_labels(i, num_classes), axis=0)))
                    if sess.run(tf.argmax(acc_hypo, axis=1))[0] == sess.run(tf.argmax(y_labels(i, num_classes), axis=0)):
                        acc_count += 1
                    # exit()
                else:
                    sess.run(optimzer, feed_dict={X:image_tensor, Y:i})

                if epoch != mem_epoch:
                    cross_entrop = sess.run(loss, feed_dict={X:image_tensor, Y:i})
                    summative_accuracy = (acc_count/total_images) * 100 if enable_accuracy else "Toggled."
                    print("Loss equitable to: {} at iteration {}. Summative accuracy: {}.".format(cross_entrop, epoch+1, round(summative_accuracy, 4)))
                    acc_count = 0
                    saver.save(sess, save_path)
                    mem_epoch = mem_epoch - 1

            i = i + 1 # NOTE: This is in the correct location
        # SAVER.save(sess, SAVE_PATH)
        # writer.add_summary(summary)

    print("\n\nTraining completed.")
    input("Press Enter to continue...")

    lol_counter = None

    if live_reply:
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

    else:
        while 1:
            DIRECTORY = input(r"Directory: ")
            test_input = sess.run(import_image(DIRECTORY))
            # print(test_input)
            hypo = sess.run(fully_connected, feed_dict={X:test_input})
            print (sorted_return(hypo, classes))

# to view tensorboard:   python -m tensorboard.main --logdir=
