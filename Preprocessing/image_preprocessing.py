import numpy as np
import cv2 as cv
import argparse
import random

from imutils import paths

# Parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print("Loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

classes = []
preprocessed_images = []

for imagePath in imagePaths:
    # read and resize image
    image = cv.imread(imagePath)
    image = cv.resize(image, (260, 260))

    # grayscale conversion, applying Gaussian blur and
    # applying Otsu's thresholding method to binarize image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(image, (5, 5), 0)
    ret, binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # removing isolated pixels by morphological erosion (kernel-based convolution)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(binary_img, kernel, iterations=1)

    # cropping images according to bounding box

