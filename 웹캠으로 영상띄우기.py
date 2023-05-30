import numpy as np
import cv2





try:
    webcam = cv2.VideoCapture(0) #웹캠을사용
    webcam.set(3,1080) #너비
    webcam.set(4,800) #높이 지정
except:
    print("웹캠 연결실패")



while(True):
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1) # 좌우 대칭
    
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