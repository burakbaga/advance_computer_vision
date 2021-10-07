import mediapipe as mp
import cv2
import time

class PoseDetection():
    def __init__(self,static_image_mode=False,model_complexity=1,smooth_landmarks=True,enable_segmentation=False,
                smooth_segmentation=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode,model_complexity,smooth_landmarks,
                                      enable_segmentation,smooth_segmentation,
                                      min_detection_confidence,min_tracking_confidence)

    def find(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(img,self.results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
        
        return img


