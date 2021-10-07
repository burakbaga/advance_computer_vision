import mediapipe as mp 
import cv2 
import time 

class HandTracking():
    def __init__(self,static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(color=(0,255,0))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode,self.max_num_hands,
                     self.min_detection_confidence,self.min_tracking_confidence)

    def find(self,img,draw=True):
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,hand_lms,self.mp_hands.HAND_CONNECTIONS)
        return img





