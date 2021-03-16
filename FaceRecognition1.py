import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    # Function detects faces and returns the cropped face
    # It returns the input image

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None

    for(x,y,w,h) in faces:                  # this loop for providing crop faces
        cropped_faces = img[y:y+h,x:x+w]

    return cropped_faces

# Initialize Webcam
cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

count=0

# Collect 100 samples of your face from webcam input
while True:
    ret, frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_name_path='G:/Faces_Sample/Face'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        # Put count on images and display live count
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('face cropper',face)

    else:
        print("Face Not Found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("All Samples Are Collected !!!!")