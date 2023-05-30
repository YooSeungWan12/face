import numpy as np
import cv2




eye_detect = False
detector = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")

#얼굴,눈인식용 xml파일
#다운 받은 파일의 경로를 적어준다.
try:
    webcam = cv2.VideoCapture(0) #웹캠을사용
    webcam.set(3,1080) #너비
    webcam.set(4,800) #높이 지정
except:
    print("웹캠 연결실패")



while(True):
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #색을 회색으로 반전

    if eye_detect:
        info = 'Eye Detection On'
    else:
        info = 'Eye Detection Off'

    faces = detector.detectMultiScale(gray,1.3, 5) #인식한걸 faces에 저장
        #얼굴이 검출된다면, 위치를 리스트로 리턴한다.  (x,y,w,h)같은 튜플로 되어있다.
    print("Number of faces detected: " + str(len(faces))) #x개 인식되었습니다 print
    cv2.putText(frame,info,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)




    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,'Detected Face',(x-5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
        if eye_detect:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            
    cv2.imshow('webcam', frame) 
        
    k = cv2.waitKey(30) & 0xff#30초동안 키입력을 기다림
    if k == ord('i'):
        eye_detect = not eye_detect
    if k == 27: # Esc 키를 누르면 종료
        break

webcam.release()

cv2.destroyAllWindows()



# https://copycoding.tistory.com/154
# 