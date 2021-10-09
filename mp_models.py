import cv2 
from modules.Hand import HandTracking
from modules.Pose import PoseDetection
from modules.Face import FaceDetectionC,FaceMeshC
import sys



def show(data,detector,name):
    cap = cv2.VideoCapture(data)
    size = int(cap.get(3)),int(cap.get(4))
    print(size)
    result = cv2.VideoWriter(f'processed_videos/{name}', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20, size)
    while True:
        ret,img = cap.read()
        if ret == True:
            img = detector.find(img)
            cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            # result.write(img)
            cv2.imshow("Image",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    
# Closes all the frames
cv2.destroyAllWindows()

def which(which_class):
    if which_class.lower()=="hand":
        detector = HandTracking()
        data = "videos/hand1.mp4"
        show(data,detector,name="handp1.avi")

    elif which_class.lower()=="pose":
        detector = PoseDetection()
        data = "videos/pose1.mp4"
        show(data,detector,name="posep1.avi")

    elif which_class.lower()=="face_detection":
        detector = FaceDetectionC()
        data = "videos/face1.mp4"
        show(data,detector,name="facedp1.avi")

    elif which_class.lower()=="face_mesh":
        detector = FaceMeshC()
        data = "videos/face3.mp4"
        show(data,detector,name="facemp1.avi")

if __name__ == "__main__":
    which_class = str(sys.argv[1])
    which(which_class)