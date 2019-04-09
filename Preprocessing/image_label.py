import os


def load_images(folder):
    # list storing image labels
    image_labels = []
    # looping through list of all image file names
    for file_name in os.listdir(folder):
        label = file_name.index("_") + 1

        # adding extracted label from image name to the list of labels
        image_labels.append(file_name[label])

    return image_labels
