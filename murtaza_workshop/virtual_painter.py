import os
import time

import cv2
import numpy as np

from hand_detector import HandDetector

folder_path = 'painter-images'
image_path_list = os.listdir(folder_path)
overlay_image_list = []
for file in image_path_list:
    img = cv2.imread(os.path.join(folder_path, file))
    img = cv2.resize(img, (1280, 128))
    overlay_image_list.append(img)
header = overlay_image_list[0]
draw_color = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(min_detection_confidence=0.85,
                        max_num_hands=1)

brush_thickness = 15
eraser_thickness = 50
# previous x and y positions of the cursor
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # read frame
    sucess, img = cap.read()
    # flip the image because the camera is flipped
    img = cv2.flip(img, 1)
    # find hand landmarks
    img = detector.find_hands(img)
    landmark_list = detector.find_lm_position(img)
    if len(landmark_list) != 0:
        # tip of index and middle fingers
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # determine which fingers are up
        fingers = detector.find_fingers_up()
        # if selection mode (two fingers are up)
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            xp, yp = x1, y1
            # check if we are in the header
            if y1 < 128:
                # select the appropriate header
                if 250 < x1 < 450:
                    header = overlay_image_list[0]
                    draw_color = (0, 0, 255)
                elif 550 < x1 < 750:
                    header = overlay_image_list[1]
                    draw_color = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlay_image_list[2]
                    draw_color = (255, 0, 0)
                elif 1050 < x1 < 1200:
                    header = overlay_image_list[3]
                    draw_color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          draw_color, cv2.FILLED)
        # if drawing mode (index finger is up only)
        elif fingers[1]:
            print("Drawing Mode")
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            # do not draw from (0, 0) to current position in the beginning
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            # draw/erase the line
            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1),
                         draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1),
                         draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1),
                         draw_color, brush_thickness)
            xp, yp = x1, y1
    
    # convert canvas to grayscale and invert it
    # only the colored region will be black
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # convert grayscale to rgb
    img_inv_rgb = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2RGB)
    # add the colored region onto the camera feed (black)
    img = cv2.bitwise_and(img, img_inv_rgb)
    # add color to the reguin that has been drawn
    img = cv2.bitwise_or(img, img_canvas)
    
    # embed header onto camera feed
    img[:128, :1280] = header
    # blend the drawing with the camera feed
    # result is transparent so ignored
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)
