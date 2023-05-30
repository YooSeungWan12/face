import numpy as np
import cv2
 
detector = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
#다운 받은 파일의 경로를 적어준다.
cap = cv2.VideoCapture(0)
 
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
 
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()