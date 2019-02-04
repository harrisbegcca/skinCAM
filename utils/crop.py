from PIL import Image
import  os

inital_file_path = "/Users/espresso/Documents/Projects/SkinCAM/input/better-smaller-database"
dest_path = "/Users/espresso/Documents/Projects/SkinCAM/input/cropped_database"

def image_cropper(current_image_path, new_image_path, number):
    image = Image.open(current_image_path)
    center_width = (image.size[0]) / 2
    center_height = (image.size[1]) / 2

    new_image = image.crop((
        center_width - 50,
        center_height - 50,
        center_width + 50,
        center_height + 50
        ))
        
    new_image.save(new_image_path+"/{}.jpg".format(number))
    # print(new_image_path+"/{}.jpg".format(number))
    return None

def file_iterator(file_path, list_images): # NOTE: list_dir is a boolean
    doublList = []
    image_folder_dirs = os.listdir(file_path)# ; print(image_folder_dirs)
    image_folder_dirs.sort()
    image_folder_dirs.remove(".DS_Store") if ".DS_Store" in image_folder_dirs else None # NOTE: keyword pass doesn't work in place of 1
    image_folder_dirs = [os.path.join(file_path, image_folder) for image_folder in image_folder_dirs]
    for folder_ind in range(len(image_folder_dirs)): #for folder in image_folder_dirs: ? - NO because I use 'i' in feed_dict={Y:i}
        os.chdir(image_folder_dirs[folder_ind])
        if list_images:
            image_names = os.listdir()
            image_names.remove(".DS_Store") if ".DS_Store" in image_names else None
            image_dir_list = [os.path.join(os.getcwd(), file_name) for file_name in image_names]
            doublList.append(image_dir_list)
        os.chdir("..")
    if list_images:
        return doublList
    else:
        return image_folder_dirs

uncropped_loc_list = file_iterator(inital_file_path, True)
# print(uncropped_loc_list)
dest_path_list = file_iterator(dest_path, False)
# print("\n", dest_path_list)

counter, celerculter = 0, 0

for uncropped_list in uncropped_loc_list:
    for uncroppped_image_path in uncropped_list:
        celerculter += 1
        # print(dest_path_list[counter])
        # print(uncroppped_image_path)
        # exit()
        # print(uncroppped_image_path)
        image_cropper(uncroppped_image_path, dest_path_list[counter], celerculter)
    counter += 1
