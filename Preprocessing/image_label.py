import os

def load_images_from_folder(folder):
    image_labels = [] #list storing image labels

    for file_name in os.listdir(folder): #looping through list of all image file names
        label = file_name.index("_") + 1
        image_labels.append(file_name[label]) #adding extracted label from image name to the list of labels

    return image_labels
