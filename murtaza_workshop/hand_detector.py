import math

import cv2
import mediapipe as mp


class HandDetector():

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.detection_conf = min_detection_confidence
        self.tracking_conf = min_tracking_confidence
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.detection_conf, self.tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

        # hand landmark results after detection
        self.results = None
        # tip ids in landmarks list
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        '''draws hand landmarks on img'''
        # mp expects RGB image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand_landmarks,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    def find_lm_position(self, img, hand_num=0, draw=False,
                         ret_bbox=False, draw_bbox=False):
        '''
        return a list containing [landmark_num, x, y] for each landmark
        for hand_num
        if ret_bbox is True, return bounding box coordinates of hand
        as (x1, y1, x2, y2)
        '''

        self.landmark_list = []
        x1, y1, x2, y2 = float('inf'), float('inf'), 0, 0
        h, w, _ = img.shape
        if self.results and self.results.multi_hand_landmarks:
            if hand_num < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_num]
                for idx, landmark in enumerate(hand.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # find bounding box
                    if cx < x1:
                        x1 = cx
                    if cy < y1:
                        y1 = cy
                    if cx > x2:
                        x2 = cx
                    if cy > y2:
                        y2 = cy
                    self.landmark_list.append([idx, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15,
                                   (255, 0, 255), cv2.FILLED)
            if draw_bbox:
                cv2.rectangle(img, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20),
                              (255, 0, 255), 2)
        if ret_bbox:
            return self.landmark_list, (x1, y1, x2, y2)
        return self.landmark_list

    def find_fingers_up(self):
        '''
        returns an length 5 array in which 
        a 1 represents a finger that is up and
        a 0 represents a finger that is down
        '''
        fingers = []
        if len(self.landmark_list) != 0:
            # determine which fingers are up
            for tip_id in self.tip_ids:
                # thumb
                if tip_id == 4:
                    # tip is to the left of pip (left hand only)
                    if self.landmark_list[tip_id][1] < self.landmark_list[tip_id - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    # comparing tip with pip
                    if self.landmark_list[tip_id][2] < self.landmark_list[tip_id - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        return fingers

    def find_distance_between_fingers(self, p1, p2, img, draw=False):
        '''
        p1 and p2 are the tips of two separate fingers
        and are in the form [landmark_num, x, y]
        '''
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # draw circle on p1
            cv2.circle(img, (x1, y1), 15,
                       (255, 0, 255), cv2.FILLED)
            # draw circle on p2
            cv2.circle(img, (x2, y2), 15,
                       (255, 0, 255), cv2.FILLED)
            # draw a line between them
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # draw a circle at the midpoint
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            
        # distance between tips of each finger
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
