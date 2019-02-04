# /*******************************************************
#  * Copyright (C) 2018 Ratnodeep Bandyopadhyay <ratnodeepb@gmail.com>
#  * 
#  * This file is part of skinCAM.
#  * 
#  * cropper can not be copied and/or distributed without the express
#  * permission of Ratnodeep Bandyopadhyay.
#  *******************************************************/

from PIL import Image
import os


PATH = os.getcwd()+"..\\input\\Database"
NEW_PATH = os.getcwd()+"..\\input\\cropped-database"
doublList = []

def crop_n_save(image_path, new_image_path, image_name):

	img = Image.open(image_path)
	mid_height = img.size[1]/2
	mid_width = img.size[0]/2

	cropped = img.crop((mid_height-50, mid_width-50, mid_height+50, mid_width+50))
	cropped.save(new_image_path+"\\{}.JPG".format(image_name))

#

image_folder_dirs = os.listdir(PATH)
image_folder_dirs.remove(".DS_Store") if ".DS_Store" in image_folder_dirs else 1
image_folder_dirs = [os.path.join(PATH, image_folder) for image_folder in image_folder_dirs]

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

i = 0

for doublconst in doublList:
	for image_dir in doublconst:
		crop_n_save(image_dir, NEW_PATH, image_name=i); i=i+1