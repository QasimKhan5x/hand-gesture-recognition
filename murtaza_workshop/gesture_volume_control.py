# Gesture Volume Control https://www.youtube.com/watch?v=9iEPzbG-xLE

import math
import time
from ctypes import POINTER, cast

import cv2
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from hand_detector import HandDetector

# set camera width and height
w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

# time used to calculate fps
previous_time = 0

# hand detector class
detector = HandDetector(min_detection_confidence=0.7, max_num_hands=1)
# area of bbox
area = 0
# how much to inc/dec volume
smoothness = 5

'''Audio control fucntions'''
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume of system (percentage of volume appearing on gui)
vol_perc = volume.GetMasterVolumeLevelScalar() * 100
# volume range of system
# volume_range = volume.GetVolumeRange()
# min_vol, max_vol = volume_range[:2]
# volume of gui
vol_bar = np.interp(vol_perc, [0, 100], [400, 150])
# color of volume label
vol_color = (255, 0, 0)

# debug
min_length, max_length = float('inf'), 0
min_area, max_area = float('inf'), 0

while True:
    # read a frame
    success, img = cap.read()
    '''Find Hand'''
    # draw hand landmarks
    img = detector.find_hands(img)
    # detect hand landmarks
    lm_list, bbox = detector.find_lm_position(
        img, ret_bbox=True, draw_bbox=True)

    if len(lm_list) != 0:
        # '''Filter based on size'''
        # calculate area of bbox
        # divide by 100 to reduce resolution
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        # debug
        # min_area = min(min_area, area)
        # max_area = max(max_area, area)
        # print('min_area=', min_area, 'max_area=', max_area)

        if 150 < area < 800:
            # '''Find distance between index and thumb'''
            # 4 = thumb tip, 8 = index tip
            thumb, index = lm_list[4], lm_list[8]
            length, img, line_info = detector.find_distance_between_fingers(
                thumb, index, img, draw=True
            )

            # debug code to find max and max lengths
            min_length = min(min_length, length)
            max_length = max(max_length, length)
            # print(f'min: {min_length}, max: {max_length}, current: {length}')

            # '''Convert length to volume'''
            '''
            Length range 70 (min) - 200 (max) (distance between thumb and index)
            Volume range -60 - 0
            convert hand length to volume range
            removed in v2 because its not smooth
            # current_volume = np.interp(length, [700, 200], [min_vol, max_vol])
            '''
            # if length is larger, then volume is larger
            # so top-left point is placed with smaller y value
            # thus range is inverted
            vol_bar = np.interp(length, [70, 210], [400, 150])
            vol_perc = np.interp(length, [70, 210], [0, 100])

            # '''reduce resolution to make it smoother'''
            # round to nearest multiple of `smoothness`
            vol_perc = smoothness * round(vol_perc / smoothness)

            # '''Check fingers up'''
            fingers = detector.find_fingers_up()

            # '''if pinky is down then set volume'''
            if not fingers[4]:
                # set system volume
                # volume.SetMasterVolumeLevel(current_volume, None)
                volume.SetMasterVolumeLevelScalar(vol_perc / 100, None)
                cv2.circle(
                    img, (line_info[-2], line_info[-1]), 15, (0, 255, 0), cv2.FILLED)
                vol_color = (0, 255, 0)
            else:
                vol_color = (255, 0, 0)
    # drawings
    # draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 255), 3)
    # fill volume bar
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400),
                  (255, 0, 255), cv2.FILLED)
    # write volume percentage
    cv2.putText(img, f'{int(vol_perc)}%', (40, 450),
                cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 3)
    # write value of current volume that has been set
    current_volume = volume.GetMasterVolumeLevelScalar() * 100
    cv2.putText(img, f'Current Vol: {int(current_volume)}',
                (350, 50), cv2.FONT_HERSHEY_DUPLEX, 1,
                vol_color, 3)

    # frame rate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
