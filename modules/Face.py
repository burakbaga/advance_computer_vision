import mediapipe as mp 
import cv2 

class FaceDetectionC():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence,model_selection)

    def find(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for detection in self.results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin*iw),int(bboxC.ymin*ih),\
                       int(bboxC.width*iw),int(bboxC.height*ih)
                if draw:
                    # img = self.fancyDraw(img,bbox)
                    cv2.rectangle(img,bbox,(255,0,255),3)
                    cv2.putText(img,f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,
                            2,(255,0,255),2)
        return img

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
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img,face_lms,self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    self.draw_spec,self.draw_spec)
                    print(self.mp_face_mesh.FACEMESH_RIGHT_EYE)
                
        return img