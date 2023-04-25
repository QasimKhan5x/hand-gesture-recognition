# Hand Tracking 30 FPS using CPU
# https://www.youtube.com/watch?v=NZde8Xt78Iw

import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
'''
This has the following parameters:
static_image_mode (bool) (default: False): 
    If true, it will perform detection all the time rather than
    detecting and tracking both.
max_num_hands (int) (default: 2):
    Maximum number of hands to detect.
min_detection_confidence (float) (default: 0.5):
    Minimum confidence value ([0.0, 1.0]) 
    for hand detection to be considered successful.
min_tracking_confidence (float) (default: 0.5):
    Minimum confidence value ([0.0, 1.0]) 
    for the hand landmarks
'''
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    # read a frame
    success, img = cap.read()
    h, w, c = img.shape
    # mp expects RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # iterate through all the hands
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                print(idx, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 15,
                           (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand_landmarks,
                                  mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
