import cv2,numpy as np
import imutils
#pip install opencv-python

def preprocessing(no): #전처리 수행 함수
    name  = "images/face/%02d.jpg"%no
    print(name)
    image = cv2.imread(name, cv2.IMREAD_COLOR) #이미지 이름저장
    # image = imutils.resize(image, width=500)
    # image = np.asarray(image, dtype=np.uint8)
    if image is None : return None, None
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #명암도 영상변환
    gray = cv2.equalizeHist(gray)#히스토그램 평활화
    return image,gray #원본영상과 명암도 영상 반환..


#얼굴 기울기 계산 함수

'''
cv2.fastAtan2(dy, dx) 함수는 OpenCV 라이브러리에서 제공되는 함수로, 주어진 y축 방향 변화량(dy)과 x축 방향 변화량(dx)을 사용하여 각도를 계산하는 함수입니다.

이 함수는 dy와 dx를 입력으로 받아, 아크탄젠트(arctangent) 함수를 이용하여 두 점 사이의 각도를 계산합니다. dy는 y축 방향의 변화량을, dx는 x축 방향의 변화량을 나타냅니다.

일반적인 math.atan2() 함수와 비슷하지만, cv2.fastAtan2()는 빠른 실행 속도를 제공하기 위해 최적화된 버전입니다. 주로 컴퓨터 비전과 이미지 처리에서 각도 계산을 수행하는 데 사용됩니다. 반환값은 라디안 단위로 표현된 각도입니다.

예를 들어, angle = cv2.fastAtan2(1, 1)은 (1, 1) 좌표에서의 각도를 계산하여 angle 변수에 저장합니다.

'''

def correct_image(image,face_center,eye_centers):
    pt0,pt1=eye_centers #좌우 눈 중심좌표
    if pt0[0] > pt1[0]: 
        pt0,pt1 = pt1 , pt0 #좌표 바꾸기.

    dx,dy = np.subtract(pt1,pt0).astype(float) # 두 위치끼리 뺄셈후 float변경
    angle = cv2.fastAtan2(dy,dx) #차분으로 기울기계산
    print(face_center)
    print(face_center[0])
    print(face_center[1])
    print(angle)
    rot_mat = cv2.getRotationMatrix2D((int(face_center[0]),int(face_center[1])),int(angle),1)#기울기만큼 회전시킴
    
    #face_center: 회전 중심점 좌표를 나타내는 튜플이나 리스트입니다. 예를 들어, (x, y) 형태의 튜플로 중심점 좌표를 전달합니다.
    #angle: 회전할 각도입니다. 양수인 경우 시계 방향으로 회전하고, 음수인 경우 반시계 방향으로 회전합니다.
    #scale: 옵션으로, 회전 후 이미지의 크기 조절 비율을 지정합니다. 기본값은 1입니다.

    size = image.shape[1::-1] #행태와 크기는 역순
    corr_image = cv2.warpAffine(image,rot_mat,size,cv2.INTER_CUBIC)

    #영상의 어파인 변환 함수 - cv2.warpAffine
    #어파인 변환 행렬을 어파인 변환 함수에 입력해주면 이동 변환을 할 수 있습니다.
    #cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst
    #src : 입력영상(변환할 이미지) 
    #rot_mat: 이미지에 적용할 내용..
    #size : 사이즈 
    #flag : 보간법  cv2.INTER_CUBIC등등..
    #https://deep-learning-study.tistory.com/175

    #cv2.INTER_CUBIC - 3차회선 보간법(4x4 이웃 픽셀 참조)
    #https://deep-learning-study.tistory.com/185

    eye_centers = np.expand_dims(eye_centers,axis=0) #첫번째 차원에 새로운축 추가
    #차원확장.. 만약(N,) 이였다면 (1,N)이 됨

    corr_centers = cv2.transform(eye_centers,rot_mat) #회전 변환 자표 계산
    corr_centers = np.squeeze(corr_centers,axis=0) #보정차원감소
    # 배열의 크기가 1인 차원을 제거하는 함수입니다.

    return corr_image,corr_centers


#입술영역 및 머리 영역 검출

def define_roi(pt,size):

    #np.ravel() 함수는 NumPy 라이브러리에서 제공되는 함수로, 
    # 다차원 배열을 1차원으로 평탄화(flatten)하는 함수입니다.

    #(pt, size) 튜플이 ([1, 2], (3, 4))라고 가정하면, 
    # np.ravel((pt, size)).astype("int")은 [1, 2, 3, 4]
    return np.ravel((pt,size)).astype("int") 


def detect_object(center,face):
    w,h = np.array(face[2:4]) #얼굴영역 크기(w,h)
    center = np.array(center)  # 얼굴 중심좌표를 ndarray객체(다차원객체)로 변경
    gap1 = np.multiply((w, h) , np.array([0.45, 0.65]))#얼굴영역 비율크기 45%,65%
    gap2 = np.multiply((w, h) , np.array([0.18, 0.1]))#입술영역 비율크기 18%,10%

    pt1 = center-gap1 #좌상단 평행이동  머리시작좌표
    pt2 = center+gap1 #우하단 평행이동 머리종료 좌표
    hair = define_roi(pt1,pt2-pt1) #전체 머리영역
    
    size = np.multiply(hair[2:4],(1,0.4)) #머리카락 영역높이 40%
    hair1 = define_roi(pt1,size) #윗머리 영역(x,y,w,h)
    hair2 = define_roi(pt2-size,size) #귀밑머리영역

    lip_center = center + (0,int(h*0.3)) #입술영역 중심좌표  - 30%
    lip1 = lip_center - gap2 #좌상단 평행이동 #입술시작좌표
    lip2 = lip_center + gap2 #우하단 평행이동 #입술종료좌표
    lip = define_roi(lip1,lip2 - lip1) #두 좌표 차분->크기 입술영역

    return [hair1,hair2,lip,hair]


