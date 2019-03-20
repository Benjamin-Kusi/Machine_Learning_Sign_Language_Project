import numpy as np
import cv2 as cv
import argparse
import random
import imutils

from imutils import paths


def find_wrist(binary_image, w, h):
    count = 0
    for j in range(w):
        b_w = binary_image[h, j]
        if b_w == 255:
            count += 1
            if count == 15:
                break
        else:
            count = 0
    return count


def crop_image(binary_image, y, x, w, h):
    crop = binary_image[y:y+h, x:x+w]
    return crop

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

    contours, _ = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100:
            continue
        #print(cv.contourArea(c))
        x, y, w, h = rect
        cv.rectangle(erosion, (x, y), (x + w, y + h), (255, 0, 0), 2)

    x, y, w, h = rect # keep track of coordinates

    # Rotating image - wrist must always be at the bottom of the frame
    wrist_row_start = h - 4
    wrist_row_end = w - 1

    # Find the wrist of the hand and rotate if you don't
    # Afterwards, crop the image at the bounding box coordinates
    ret = find_wrist(erosion, wrist_row_end, wrist_row_start)
    if ret != 15:
        # image has no wrist, rotate it 270 degrees
        erosion = imutils.rotate(erosion, 270)
        cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

    else:
        # wrist found
        cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

    cv.imshow("Cropped", cropped)
