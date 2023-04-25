import time

import autopy
import cv2
import numpy as np

from hand_detector import HandDetector

cap = cv2.VideoCapture(0)
cam_width, cam_height = 640, 480
cap.set(3, cam_width)
cap.set(4, cam_height)
screen_width, screen_height = autopy.screen.size()

previous_time = 0
# frame reduction
frame_r = 100
smoothening = 7
prev_x, prev_y = autopy.mouse.location()
curr_x, curr_y = autopy.mouse.location()

detector = HandDetector(min_detection_confidence=0.85,
                        max_num_hands=1)

# debug
# min_length = float('inf')

while True:
    # read frame
    sucess, img = cap.read()
    # find hand landmarks
    img = detector.find_hands(img)
    landmark_list = detector.find_lm_position(img)

    # draw a rectangle to mark the area for moving the mouse
    cv2.rectangle(img, (frame_r, frame_r),
                  (cam_width - frame_r, cam_height - frame_r),
                  (255, 0, 255), 2)

    # get the tip of the index and middle fingers
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # check which fingers are up
        fingers = detector.find_fingers_up()

        # only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # convert coordinates
            x3 = np.interp(x1, (frame_r, cam_width - frame_r),
                           (0, screen_width))
            y3 = np.interp(y1, (frame_r, cam_height - frame_r),
                           (0, screen_height))

            # smoothen values
            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening
            # update previous location of pointer
            prev_x, prev_y = curr_x, curr_y

            # move mouse (flip x axis to match the screen)
            # prevent out of bounds
            x3_flip = min(screen_width - x3, screen_width - 0.01)
            y3 = min(y3, screen_height - 0.01)
            autopy.mouse.move(x3_flip, y3)
            # draw a circle on the index finger when moving
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # if index and middle finger are up: clicking mode
        elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            length, img, line_info = detector.find_distance_between_fingers(
                landmark_list[8], landmark_list[12], img, draw=True
            )

            # min_length = min(min_length, length)
            # print(min_length)

            # if thumb and mid are close, click the mouse
            if length < 30:
                cv2.circle(
                    img, (line_info[-2], line_info[-1]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()
    # frame rate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
