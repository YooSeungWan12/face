import numpy as np
import cv2
 
detector = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
#얼굴인식용 xml파일
#다운 받은 파일의 경로를 적어준다.
try:
    webcam = cv2.VideoCapture(0) #웹캠을사용
    webcam.set(3,1080) #너비
    webcam.set(4,800) #높이 지정



    while(True):
        ret, frame = webcam.read()
        frame = cv2.flip(frame, 1) # 좌우 대칭
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #색을 회색으로 반전


        faces = detector.detectMultiScale(gray,1.05, 5) #인식한걸 faces에 저장
        #얼굴이 검출된다면, 위치를 리스트로 리턴한다.  (x,y,w,h)같은 튜플로 되어있다.
        print("Number of faces detected: " + str(len(faces))) #x개 인식되었습니다 print

        if len(faces):#존재한다면..
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        cv2.imshow('webcam', frame) 
        
        k = cv2.waitKey(30) & 0xff#30초동안 키입력을 기다림
        if k == 27: # Esc 키를 누르면 종료
            break

    webcam.release()

    cv2.destroyAllWindows()

except:
    print("웹캠 연결실패")


# https://copycoding.tistory.com/154
# 