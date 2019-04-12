import math
import cv2 as cv
import numpy as np
import imutils


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


def find_finger_tip(binary_image, x, y, w, h):
    x_coords = []
    y_coords = []
    finger_count = 0
    pixel_counter = 0

    # iterate over the image in x and y
    for j in range(y, h):
        for i in range(x, w):
            pixel = binary_image[i][j]
            # is the pixel  white ?
            if pixel == 255:
                pixel_counter += 1
                # white pixel counter checks for 3 consecutive white pixels
                if pixel_counter == 12:
                    if check_left_right(binary_image, i, pixel_counter, j):
                        if j == y:
                            if check_bottom(binary_image, i , j):
                                finger_count += 1
                            else:
                                pixel_counter = 0
                        else:
                            if check_top_bottom(binary_image, i, j):
                                x_coords.append(i)
                                y_coords.append(j)
                                finger_count += 1
                            else:
                                pixel_counter = 0
                    else:
                        pixel_counter = 0
            else:
                pixel_counter = 0
    return [finger_count, x_coords, y_coords]


# check for left and right of three continuous pixels
def check_left_right(binary_image, i, p_counter, j):
    status = False
    left_pixel = i - p_counter
    right_pixel = i + 1

    r = binary_image[right_pixel][j]
    l = binary_image[left_pixel][j]

    if binary_image[right_pixel][j] == 0 and binary_image[left_pixel][j] == 0:
        status = True
    print("left right status", status)
    return status


# checking bottom if on top row
def check_bottom(binary_image, i_val, j_val):
    bottom_row = j_val + 1
    status = False

    a2 = binary_image[i_val][bottom_row]
    b2 = binary_image[i_val - 1][bottom_row]
    c2 = binary_image[i_val - 2][bottom_row]

    sum_bottom = a2 + b2 + c2
    if sum_bottom == 765:
        status = True
    else:
        status = False
    return status


# check if top pixels are black or single white pixel
# and bottom pixels are white
def check_top_bottom(binary_image, i_val, j_val):
    bottom_row = j_val + 1
    top_row = j_val - 1
    status = False

    a1 = binary_image[i_val][top_row]
    a2 = binary_image[i_val][bottom_row]
    b1 = binary_image[i_val - 1][top_row]
    b2 = binary_image[i_val - 1][bottom_row]
    c1 = binary_image[i_val - 2][top_row]
    c2 = binary_image[i_val - 2][bottom_row]

    sum_top = a1 + b1 + c1
    sum_bottom = a2 + b2 + c2
    # top condition
    if sum_top == 0 or sum_top == 255:
        if sum_bottom == 765:
            status = True
        else:
            status = False
    else:
        status = False
    return status


def find_centroid(x_coords, y_coords):
    n = len(x_coords)
    x_coord = sum(x_coords)/n
    y_coord = sum(y_coords)/n

    centroid = x_coord, y_coord

    return centroid


def calc_angle_distance(centroid, vertice_x, vertice_y):
    angle_list = []
    distance_list = []

    for i in range(len(vertice_x)):
        dx = centroid[0] - vertice_x[i]
        dy = centroid[1] - vertice_y[i]
        rad = math.atan2(dx/dy)
        deg = math.degrees(rad)
        angle_list.append(deg)

        distance_list.append(math.sqrt((vertice_x[i] - centroid[0])**2 + (vertice_y[i] - centroid[1])**2))

    return angle_list, distance_list


def calc_area(binary_image, x, y, w, h):
    white_pixels = 0
    for i in range(x, w):
        for j in range(y, h):
            if binary_image[i][j] == 255:
                white_pixels += 1
            else:
                continue

    return white_pixels

image = cv.imread("/Users/lvz/PycharmProjects/Machine_Learning_Sign_Language_Project/Preprocessing/test3.png")
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

#ret = find_wrist(erosion, wrist_row_end, wrist_row_start)
#print(ret)
#if ret != 50:
    # image has no wrist, rotate it 90 degrees
  #  print("rotating")
 #   erosion = imutils.rotate(erosion, 270)
   # cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

#else:
    # wrist found
 #   print("found wrist")
  #  cropped = crop_image(erosion, y, x, wrist_row_end, wrist_row_start)

#cv.imshow("Original", binary_img)
#cv.imwrite("Original.png", binary_img)
#cv.imshow("Erosion", erosion)

#cv.imwrite("Erosion.png", erosion)
#cv.imshow("Rotated & Cropped", erosion)

#cv.imwrite("Final.png", cropped)
#cv.imshow("Last", cropped)
#cv.waitKey(0)
#cv.destroyAllWindows()

num = find_finger_tip(erosion, x, y, w, h)
print(num)
#calc_area(erosion, x, y, w, h)
