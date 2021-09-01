import cv2 as cv
import time
import numpy as np
import handTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


######## PARAMETERS ########
wCam, hCam = 640, 480
############################
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
cTime = 0

detector = htm.handDetector(detectionConfidence=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0
cVol = int(volume.GetMasterVolumeLevelScalar()*100)

volColor = (255,0,0)

while True:
    success, img = cap.read()

    #Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img,draw=[4,8],boundingBoxDist=0)
    if len(lmList)!=0:

        #Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])//100
        #print(area)
        if 250<area<1750:

            #Find distance between index and thumb
            length, img, points = detector.findDistance(4,8,img,True)

            #Convert volume
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            #Reduce resolution to make it smoother
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)

            #Check fingers up
            fingers = detector.fingersUp()

            #If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cVol = int(volume.GetMasterVolumeLevelScalar()*100)
                cv.circle(img, (points[4], points[5]), 10, volColor, cv.FILLED)
                volColor = (0,255,0)
            else:
                volColor = (0,255,0)

    #Drawings
            cv.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
    cv.rectangle(img, (50,int(volBar)), (85,400), (200,0,0),cv.FILLED)
    cv.rectangle(img, (50,150), (85,400), (255,0,0),3)
    cv.putText(img,f'{int(volPer)}%', (30, 440), cv.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 3)
    cv.putText(img,f'Vol. Set: {cVol}', (400, 40), cv.FONT_HERSHEY_PLAIN, 2, volColor, 3)

    #FrameRate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}', (10, 40), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)