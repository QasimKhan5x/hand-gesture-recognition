# Hand Tracking 30 FPS using CPU
# https://www.youtube.com/watch?v=NZde8Xt78Iw

import time

import cv2
from hand_detector import HandDetector


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    previous_time = 0
    current_time = 0

    while True:
        # read a frame
        success, img = cap.read()
        img = detector.find_hands(img)
        landmark_list = detector.find_lm_position(img, draw=False)
        # print 4th mark position
        if len(landmark_list) != 0:
            lm = landmark_list[4]
            print(f"x: {lm[1]}, y: {lm[2]}")
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
