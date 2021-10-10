import mediapipe as mp 
import cv2
import numpy as np 
from skimage.draw import polygon
from modules.FaceMesh import FaceMeshC

DATA = "videos/face1.mp4"

def rescale_frame(frame,scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dims = (width,height)

    return cv2.resize(frame,dims,interpolation=cv2.INTER_AREA)

detector = FaceMeshC()
cap = cv2.VideoCapture(DATA)
SIZE = int(cap.get(3)),int(cap.get(4))
result = cv2.VideoWriter(f'processed_videos/{"deneme3.avi"}', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20, SIZE)
while True:
    ret,img = cap.read()
    img_original = img.copy()
    if ret == True:
        img,poly = detector.find(img)
        try : 
            Y, X = polygon(poly[:,1], poly[:,0])
        except:
            pass

        cropped_img = np.zeros(img.shape, dtype=np.uint8)
        cropped_img[Y, X] = img_original[Y, X]
        cropped_img = rescale_frame(cropped_img,scale=.85)
        zeros = np.zeros((img.shape))

        zeros = np.zeros((img.shape),np.uint8)
        img_height = img.shape[0]
        img_width = img.shape[1]
        cropped_img_height = cropped_img.shape[0]
        cropped_img_width = cropped_img.shape[1]

        zeros[img_height-cropped_img_height:img_height,img_width-cropped_img_width:img_width] = cropped_img

        img_gray = cv2.cvtColor(zeros,cv2.COLOR_BGR2GRAY)
        _,img_inv =cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,img_inv)
        img = cv2.bitwise_or(img,zeros)

        result.write(img)
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

cv2.destroyAllWindows()
