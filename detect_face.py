import cv2,numpy as np
#pip install opencv-python

def preprocessing(no): #전처리 수행 함수
    name  = "images/face1/%2d.jpg"%no
    print(name)
    image = cv2.imread(name, cv2.IMREAD_COLOR) #이미지 이름저장
    print(image)
    # image = np.asarray(image, dtype=np.uint8)
    if image is None : return None, None
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #명암도 영상변환
    gray = cv2.equalizeHist(gray)#히스토그램 평활화
    return image,gray #원본영상과 명암도 영상 반환..


face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")

image,gray = preprocessing(15)
if image is None:
    raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray,1.1,2,0,(100,100)) #얼굴검출
#detectMultiScale(gray,영상확대비율 ,이웃후보개수,0,최소얼굴크기)

print("face : ",faces)
if faces.any():#얼굴 검출되면
    x,y,w,h = faces[0] #검출사각형
    face_image = image[y:y+h,x:x+w] #얼굴 영역 영상 가져오기
    # eyes = eye_cascade.detectMultiScale(face_image,1.15,7,0,(25,20))#눈검출
    # if len(eyes) == 2: #눈 사각형이 검출되면
    #     for ex,ey,ew,eh in eyes:
    #         center = (x+ex+ew//2,y+ey+eh//2) #중심점 계산
    #         cv2.circle(image,center,10,(0,255,0),2) #눈 중심에 원그리기
    # else:
    #     print("눈 미검출")

    cv2.rectangle(image,faces[0],(255,0,0),2) #얼굴 검출 사각형 그리기
    cv2.imshow("image",image)
else:
    print("얼굴 미검출")
cv2.waitKey()
    

