import os
import time

import cv2

from hand_detector import HandDetector

cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

folder_path = "hand-images"
image_path_list = os.listdir(folder_path)
# overlay the fingers on the camera feed
overlay_image_list = []
for image_path in image_path_list:
    image = cv2.imread(os.path.join(folder_path, image_path))
    # image = cv2.resize(image, (128, 128))
    overlay_image_list.append(image)

previous_time = 0
detector = HandDetector(min_detection_confidence=0.8, max_num_hands=1)
tip_ids = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    img = detector.find_hands(img)
    landmark_list = detector.find_lm_position(img)

    fingers = []
    if len(landmark_list) != 0:
        # determine which fingers are up
        for tip_id in tip_ids:
            # thumb
            if tip_id == 4:
                # tip is to the left of pip
                if landmark_list[tip_id][1] < landmark_list[tip_id - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # comparing tip with pip
                if landmark_list[tip_id][2] < landmark_list[tip_id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
    total_fingers = sum(fingers)
    # embed overlay onto camera feed
    h, w, c = overlay_image_list[total_fingers].shape
    img[:h, :w] = overlay_image_list[total_fingers]
    # display total fingers
    cv2.rectangle(img, (0, 235), (150, 415),
                  (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(total_fingers), (25, 385),
                cv2.FONT_HERSHEY_PLAIN,
                10, (255, 0, 0), 25)
    # calculate fps
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, str(int(fps)), (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
