import cv2 as cv
import time
import handTrackingModule as htm

wCam, hCam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionConfidence=0.7)

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=[a for a in range(21)])

    totalFingers = 0
    if len(lmList) != 0:
        fingers = []
        #THUMB
        if lmList[4][1] > lmList[0][1]:
            if lmList[4][1]>lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[4][1]<lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        #OTHER FINGERS
        for i in range(8,21,4):
            if lmList[i][2]<lmList[i-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}', (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv.rectangle(img, (20,280),(130,420),(0,255,0),cv.FILLED)
    cv.putText(img,str(totalFingers), (25, 400), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 15)

    cv.imshow('Image', img)
    cv.waitKey(1)