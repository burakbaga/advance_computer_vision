import mediapipe as mp 
import cv2
import numpy as np 
from skimage.draw import polygon


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
                    # print(self.mp_face_mesh.FACEMESH_FACE_OVAL)
                    # print(list(self.mp_face_mesh.FACEMESH_FACE_OVAL))
                    # points = list(self.mp_face_mesh.FACEMESH_FACE_OVAL)
                    # liste = []
                    # liste.append(points[0][0])
                    # liste.append(points[0][1])
                    # search = points[0][1]
                    # points.remove((points[0][0],search))

                    # def search(points,search):
                    #     for iw in points:
                    #         if search == iw[0]:
                    #             liste.append(iw[1])
                    #             points.remove((search,iw[1]))
                    #             search = iw[1]

                    
                    # while len(liste)<34:
                    #     search(points,search)
                    #     print(liste)
                    sorted_lm = [176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176]
                    poly = []
                    # print(list(self.mp_face_mesh.FACEMESH_FACE_OVAL))
                    points = {}
                    for id,lm in enumerate(face_lms.landmark):
                        ih,iw,ic = img.shape
                        x,y = int(lm.x*iw),int(lm.y*ih)
                        if id in sorted_lm:
                            # cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
                            points[f"{id}"] = [x,y]
                    for id in sorted_lm:
                        if points:
                            poly.append(points[f"{id}"])

                        
                        

                    # print(ids)

        return img,np.array(poly)

def rescale_frame(frame,scale=0.50):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dims = (width,height)

    return cv2.resize(frame,dims,interpolation=cv2.INTER_AREA)
detector = FaceMeshC()

detector = FaceMeshC()
data = "videos/face11.mp4"
cap = cv2.VideoCapture(data)
size = int(cap.get(3)),int(cap.get(4))
print(size)
result = cv2.VideoWriter(f'processed_videos/{"deneme2.avi"}', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         20, size)
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

        # cv2.rectangle(img,(img_width-cropped_img_width,img_height-cropped_img_height),(img_width,img_height),(0,0,255),10)

        # zeros[0:cropped_img.shape[0],0:cropped_img.shape[1]] = cropped_img

        img_gray = cv2.cvtColor(zeros,cv2.COLOR_BGR2GRAY)
        _,img_inv =cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,img_inv)
        img = cv2.bitwise_or(img,zeros)



  
  

        # img[0:200,0:300] = cropped_img


        result.write(img)
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
# result.release()
# Closes all the frames
cv2.destroyAllWindows()



# image = cv2.imread("images/face.jpg")
# image,poly = detector.find(image)
# print("----------------------------------------")
# print(poly)
# print(image.shape)


# Y, X = polygon(poly[:,1], poly[:,0])
# cropped_img = np.zeros(image.shape, dtype=np.uint8)
# cropped_img[Y, X] = image[Y, X]


# cv2.imshow("Image",image)
# cv2.imshow("Cropped",cropped_img)


# cv2.waitKey(0)