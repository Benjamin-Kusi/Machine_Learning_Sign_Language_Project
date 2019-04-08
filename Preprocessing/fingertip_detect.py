import cv2 as cv

# iterates through cropped image to find fingertips

def find_finger_tip(binary_image, x, y, w, h):
    finger_count = 0
    pixel_counter = 0

    for j in range(y, w):
        top_row = 0
        white_pixel = binary_image[top_row, j-1]

        if white_pixel == 255:
            pixel_counter += 1

            if pixel_counter == 3:

                if check_left_right(j, pixel_counter):
                    if check_top_bottom(binary_image, x, y, w, h):
                        finger_count += 1
                        find_finger_tip(binary_image, x, y, w, h)
                    #finger_count = check_top_bottom(binary_image, x, y, w, h)

                else:
                    y += 1

                    if y < h:
                        find_finger_tip(binary_image, x, y, w, h)

        else:
            pixel_counter = 0

    return finger_count


def check_left_right(j, pixel_counter):
    status = False

    left_pixel = j - pixel_counter
    right_pixel = j + 1

    if right_pixel != 255 and left_pixel != 255:
        status = True

    return status


def check_top_bottom(binary_image, x, y, w, h):
    bottom_pixel = y + 1
    top_pixel = y - 1
    status = False

    # check if top is black and bottom is white
    if top_pixel != 255 and bottom_pixel == 255:
        status = True
        #finger_count += 1
        #find_finger_tip(binary_image, x, y, w, h)

    return status
