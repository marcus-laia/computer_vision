import cv2 as cv
import numpy as np
import handTrackingModule as htm
import time
import win32api, win32con

wCam, hCam = 640, 480
wScreen, hScreen = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
frameRed = 100 #Frame Reduction
smoothening = 5

plocX, plocY = 0,0 #Previous Location
clocX, clocY = 0,0 #Current Location

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(maxHands=1,detectionConfidence=0.7)


while True:
    success, img = cap.read()

    # 1. Find Hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img,draw=[i for i in range(21)],markSize=0,boundingBoxDist=0)

    # 2. Get the tip of the index
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]

        # 3. Select part of the video to represent the screen
        fingers = detector.fingersUp()
        cv.rectangle(img,(frameRed, frameRed),(wCam-frameRed, hCam-frameRed), (255,0,255), 2)

        # 4. Index Finger Up = Moving Mode
        # if the pointer is flicking when clicking, add a possibility in which moving stop
        if fingers[1]==1 and fingers[2]==1:

            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameRed,wCam-frameRed),(0,wScreen))
            y3 = np.interp(y1, (frameRed,hCam-frameRed),(0,hScreen))

            # 6. Smoothen Values
            clocX = plocX+(x3-plocX)/smoothening
            clocY = plocY+(y3-plocY)/smoothening

            # 7. Move Mouse
            win32api.SetCursorPos((wScreen-int(clocX),int(clocY)))
            cv.circle(img,(x1,y1), 8, (255,0,255), cv.FILLED)
            plocX, plocY = clocX, clocY

            cv.putText(img,'CURSOR UNLOCKED', (20, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv.putText(img,'CURSOR LOCKED', (20, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # 9. Find distance between fingers
        lenght, img, lineInfo = detector.findDistance(8, 12, img)
        if lenght < 60:
            # 8. Thumb close = Click
            if fingers[0]==0:
            # 10. Click mouse if distance is short
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
                time.sleep(0.2)

            cv.putText(img,'CLICK UNLOCKED', (20, 420), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv.putText(img,'CLICK LOCKED', (20, 420), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # 12. Display
    imgcut = cv.flip((img[frameRed:hCam-frameRed+1,frameRed:wCam-frameRed+1]), 1)
    imgcut = cv.resize(imgcut,(imgcut.shape[1]//1,imgcut.shape[0]//1))

    #cv.imshow('ImageCut', imgcut)
    cv.imshow('Image', img)
    cv.waitKey(1)