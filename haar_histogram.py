import cv2,numpy as np


#타원 그리기 함수
def draw_ellipse(image,roi,ratio,color,thickness=cv2.FILLED):
    x, y, w, h = roi
    center = (x + w//2 , y + h//2)
    size = (int(w*ratio), int(h*ratio)) #그려질 타원비율
    cv2.ellipse(image,center,size,0,0,360,color,thickness)
    #타원을 그리는 함수
    #image : 타원을 그릴 이미지
    #center : 중심좌표  , 튜플형식
    #size : 장축과 단축의 반지름길이, 튜플형식
    #angle : 회전각도  0
    #startAngle : 타원시작각도  0도는 3시방향
    #endAngle : 타원의 종료각도 360도는 한바퀴
    #color:색상,  (B,G,R)같은 튜플형식
    #thickness 타원의 선 두께   음수값주면 내부를 채운다고함..
    return image


#각 마스크 생성함수

def make_masks(rois,shape):#영역별 마스크생성함수
    base_mask = np.full(shape,255,np.uint8)
    hair_mask = draw_ellipse(base_mask,rois[3],0.45,0) #얼굴타원그리기
    lip_mask = draw_ellipse(np.copy(base_mask),rois[2],0.40,255)#입력타원그리기
    #기본 마스크에 입술영역 타원그리기


    masks = [hair_mask,hair_mask,lip_mask,~lip_mask] #4개 마스크생성, ~lip은 입술마스크 반전
    masks = [mask[y:y+h,x:x+w] for mask, (x,y,w,h) in zip(masks,rois)]
    # for mask, (x,y,w,h) in zip(masks,rois): masks와 rois 리스트를 동시에 순회하면서 각각의 요소들을 mask와 (x,y,w,h)로 언패킹합니다.
    # mask[y:y+h,x:x+w]: mask에서 (x,y) 좌표부터 (x+w,y+h) 좌표까지의 부분 영역을 추출하여 새로운 마스크 이미지를 생성합니다.
    # 이렇게 생성된 마스크 이미지를 리스트 컴프리헨션으로 생성된 masks 리스트에 추가합니다.
    # 결과적으로, masks 리스트에 각 관심영역에 해당하는 마스크이미지들이 저장됨..

    # for i ,mask in enumerate(masks):  마스크 영상 윈도우표시
    #     cv2.imshow('mask'+str(i),mask)
    # cv2.waitKey()

    return masks



#마스크를 이용하여 각 서브영역의 히스토그램을 생성
#기울기 보정 영상에 각 서브 영역 및 마스크로 히스토그램 계산

def calc_histo(image,rois,masks):
    bsize = (64,64,64) #히스토그램 계급갯수?? 잘안보임..
    ranges = (0,256,0,256,0,256) #각 채널 빈도범위
    
    #해당영역 참조영상
    subs = [image[y:y+h,x:x+w] for x,y,w,h in rois] #관심영역 참조로 영상생성

    #관심영역 영상 히스토그램
    hists = [cv2.calcHist([sub],[0,1,2],mask,bsize,ranges,3) for sub,mask in zip(subs,masks)]
    #masks는 각 영역 마스크 4개..

    #히스토그램 정규화
    hists = [h/np.sum(h) for h in hists]

    #입술-얼굴 유사도
    sim1 = cv2.compareHist(hists[2],hists[3],cv2.HISTCMP_CORREL)
    #cv2.compareHist() 함수는 두 개의 히스토그램을 비교하여 유사성을 측정하는 기능
    #hists[2]와 hists[3]는 비교할 두 개의 히스토그램
    #cv2.HISTCMP_CORREL은 비교 방법을 지정하는 상수로, 상관 계수(correlation)를 사용하여 히스토그램의 유사도를 계산하도록 설정됩니다.
    #sim1은 hists[2]와 hists[3] 사이의 유사도를 나타내는 값으로, 
    #일반적으로 -1에서 1 사이의 범위를 가집니다. 
    # 유사도가 높을수록 값은 1에 가까워지며, 유사도가 낮을수록 값은 -1에 가까워집니다

    #윗-귓밑머리 유사도
    sim2 = cv2.compareHist(hists[0],hists[1],cv2.HISTCMP_CORREL)

    return sim1,sim2