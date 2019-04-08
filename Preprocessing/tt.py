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
    finger_count = 0
    pixel_counter = 0
    print("y is, w is", y, w)

    # iterate over the row from y to w
    for j in range(y, w):
        top_row = 0
        white_pixel = binary_image[top_row, j-1]
        print("j, top row, white pixel", j, top_row, white_pixel)

        # is the white pixel actually white ?
        if white_pixel == 255:
            pixel_counter += 1
            print("pixel counter", pixel_counter)

            # white pixel counter checks for 3 consecutive white pixels
            if pixel_counter == 3:

                if check_left_right(j, pixel_counter):
                    if check_top_bottom(binary_image, x, y, w, h,j):
                        print("top bottom checked")
                        finger_count += 1
                        print("2. finger count is", finger_count)
                        find_finger_tip(binary_image, x, y, w, h)
                    #finger_count = check_top_bottom(binary_image, x, y, w, h)

                else:
                    print("nyet")
                    y += 1

                    if y < h:
                        find_finger_tip(binary_image, x, y, w, h)

            # counter isn't 3, what to do?


        # it's not white, move to next pixel
        else:
            pixel_counter = 0
            print("pixel counter", pixel_counter)
            #continue

    return finger_count


# check for left and right of three continuous pixels
def check_left_right(j, pixel_counter):
    status = False
    left_pixel = j - pixel_counter
    right_pixel = j + 1

    print("left, right", left_pixel,right_pixel)
    if right_pixel != 255 and left_pixel != 255:
        status = True
    print("status", status)
    return status


# check if top pixels are black or single white pixel
# and bottom pixels are white
def check_top_bottom(binary_image, x, y, w, h,j):
    bottom_pixel = y + 2
    top_pixel = y
    status = False

    print("top and bottom pixels", top_pixel,bottom_pixel)
    print("top pixel val", binary_image[j][top_pixel])
    print("bottom pixel val", binary_image[j][bottom_pixel])

    # check if top is black or single white pixel then, and if bottom pixels are white
    if binary_image[j][top_pixel] != 255 and binary_image[j][bottom_pixel] == 255:
        print("bottom pixels are white")
        status = True

    print("top/bottom status", status)
    return status


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
    print(x, y, w, h)
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
#cv.imwrite("Original.png", binary_img)
#cv.imshow("Erosion", erosion)

#cv.imwrite("Erosion.png", erosion)
#cv.imshow("Rotated & Cropped", cropped)

#cv.imwrite("Final.png", cropped)
#cv.imshow("Last", cropped)
#cv.waitKey(0)
#cv.destroyAllWindows()

num = find_finger_tip(cropped, x, y, w, h)
#print(num)
