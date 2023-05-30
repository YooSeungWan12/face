from harr_utils import *
from haar_histogram import *


face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")


image,gray = preprocessing(34)
if image is None:
    raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray,1.1,2,0,(100,100)) #얼굴검출
#detectMultiScale(gray,영상확대비율 ,이웃후보개수,0,최소얼굴크기)

print(faces)
if faces.any():#얼굴 검출되면
    x, y, w, h = faces[0] #검출사각형
    face_image = image[y:y+h,x:x+w] #얼굴 영역 영상 가져오기
    eyes = eye_cascade.detectMultiScale(face_image,1.15,7,0,(25,20))#눈검출
    if len(eyes) == 2: #눈 사각형이 검출되면
        face_center =(x+w//2,y+h//2)
        eye_centers = [(x+ex+ew//2,y+ey+eh//2) for ex,ey,ew,eh in eyes]
        corr_image,corr_center = correct_image(image,face_center,eye_centers)#회전 보정
        
        print(corr_image)
        print(corr_image.shape[:2])

        rois = detect_object(face_center,faces[0]) #머리카락,입술영역 계산
        masks = make_masks(rois,corr_image.shape[:2])
        sim = calc_histo(corr_image,rois,masks)

        print("윗-귀밑머리유사도: %4.2f"%sim[0])
        print("입술-얼굴 유사도: %4.2f"%sim[1])

    else:
        print("눈 미검출")


else:
    print("얼굴 미검출")

cv2.imshow("image",image)
cv2.waitKey(0)