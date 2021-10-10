import mediapipe as mp 
import cv2 
import numpy as np 


class FaceMeshC():
    def __init__(self,
            static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5,
            min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_image_mode,self.max_num_faces,
                                                    self.min_detection_confidence,self.min_tracking_confidence)
        self.draw_spec = self.mp_drawing.DrawingSpec(thickness=1,circle_radius=2,color=(0,255,0))
        self.mp_drawing_styles = mp.solutions.drawing_styles
    def find(self,img,draw=True):
        poly = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,face_lms,self.mp_face_mesh.FACEMESH_TESSELATION,
                    self.draw_spec,self.draw_spec)
                    sorted_lm = [176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176]
                    poly = []
                    points = {}
                    for id,lm in enumerate(face_lms.landmark):
                        ih,iw,ic = img.shape
                        x,y = int(lm.x*iw),int(lm.y*ih)
                        if id in sorted_lm:
                            points[f"{id}"] = [x,y]
                    for id in sorted_lm:
                        if points:
                            poly.append(points[f"{id}"])

        return img,np.array(poly)