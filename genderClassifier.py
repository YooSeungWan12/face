from harr_utils import * 
from haar_histogram import *
from haar_classify import *


face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")

no,max_no,cnt= 0,61,1

while True:
    no = no + cnt
    image,gray = preprocessing(no)#전처리 수행

    if image is None: #예외처리
        print("%02d.jpg :영상 파일없음"% no)
        if no < 0 : 
            no = max_no
        elif no >= max_no : 
            no=0
        continue


    faces = face_cascade.detectMultiScale(gray,1.1,2,0,(100,100))#얼굴검출
    print("face: ",faces)
    if faces.any():
        x,y,w,h = faces[0]
        face_image = image[y:y+h, x:x+w] #얼굴영역 가져오기
        eyes = eye_cascade.detectMultiScale(face_image,1.15,7,0,(25,20))

        if len(eyes) == 2:
            face_center = (x+w//2,y+h//2)
            eye_centers = [(x+ex+ew//2 , y+ey+eh//2) for ex,ey,ew,eh in eyes]
            corr_image, corr_centers = correct_image(image,face_center,eye_centers)

            rois = detect_object(face_center , faces[0])#4개 영역계산
            masks = make_masks(rois,corr_image.shape[:2]) #각 영역 마스크 생성
            sims = calc_histo(corr_image,rois,masks) #히스토그램 생성

            classify(corr_image,sims,no) #성별 분류 및 표시
            display(corr_image,face_center,corr_centers , rois)#얼굴과 눈 표시

        else : 
            print("%02d.jpg: 눈 미검출"%no)
    else:
        print("%02d.jpg : 얼굴 미검출"%no)

    key = cv2.waitKeyEx(0) #키 이벤트 대기
    if key == 0x260000 : #위 화살표 
        cnt = 1
    elif key == 0x280000: #아래화살표
        cnt = -1
    elif key == 32 or key == 27: #32: 스페이스  27 :esc
        break
    else:
        cnt=0

            

