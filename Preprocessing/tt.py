import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt

def crop_image(binary_image, y, x, w, h):
    crop = binary_image[y:y+h, x:x+w]
    return crop

def find_wrist(binary_image, w, h):
    count = 0
    for j in range(w):
        b_w = binary_image[h, j]
        if b_w == 255:
            count += 1
            if count == 50:
                break
        else:
            count = 0
    return count

def find_finger(binary_image, y, x, w, h):
    cord_dict = {}
    top_row = x
    left_col = y

    bottom_most_row = h
    right_col = w

    finger_count = 0

    if top_row <= bottom_most_row:
        for j in range(y, w):
            b_w = binary_image[x, j]

    else:
        return finger_count




image = cv.imread("/Users/lvz/PycharmProjects/Machine_Learning_Sign_Language_Project/Preprocessing/test2.png")
image = cv.resize(image, (360, 360))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(image, (5, 5), 0)
ret, binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)

erosion = cv.erode(binary_img, kernel, iterations=1)
#cv.imshow("Original", binary_img)
#cv.imshow("Show", erosion)
#cv.waitKey()

contours, _ = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    #print("found")
    rect = cv.boundingRect(c)
    if rect[2] < 100 or rect[3] < 100:
        continue
    #print(cv.contourArea(c))
    x, y, w, h = rect
    #print(x, y, w, h)
    #cv.rectangle(erosion, (x, y),(x+w, y+h),(255, 0, 0), 2)
x, y, w, h = rect
wrist_row_start = h - 4
wrist_row_end = w - 1

ret = find_wrist(erosion, wrist_row_end, wrist_row_start)
print(ret)
if ret != 50:
    # image has no wrist, rotate it 90 degrees
    print("rotating")
    erosion = imutils.rotate(erosion, 270)
    cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

else:
    # wrist found
    print("found wrist")
    cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

#cv.imshow("Original", binary_img)
cv.imwrite("Original.png", binary_img)
#cv.imshow("Erosion", erosion)

cv.imwrite("Erosion.png", erosion)
#cv.imshow("Rotated & Cropped", cropped)

cv.imwrite("Final.png", cropped)
cv.waitKey(0)
cv.destroyAllWindows()
