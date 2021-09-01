import cv2 as cv
import numpy as np

frameWidth = 640
frameHeight = 480

cap = cv.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
#cap.set(10,100)

myColors = [[88,78,0,150,255,255],
            [0,132,128,49,255,255],
            [32,72,0,101,255,255]]

colorValues = [[255,0,0],
               [0,127,255],
               [0,255,0]]

myPoints = [] # [x,y,colorID]

def findColor(img, myColors):
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    newPoints = []
    for color in myColors:
        lower = np.array([color[0:3]])
        upper = np.array([color[3:6]])
        mask = cv.inRange(imgHSV, lower, upper)
        x,y = getContours(mask)
        cv.circle(imgResult,(x,y),10,colorValues[myColors.index(color)],cv.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,myColors.index(color)])
        #cv.imshow(str(myColors.index(color)), mask)
    return newPoints


def getContours(img):
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>300:
            #cv.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv.boundingRect(approx)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return x+w//2,y

def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv.circle(imgResult,(point[0],point[1]),10, myColorValues[point[2]],cv.FILLED)

while True:
    success, img = cap.read()

    imgResult = img.copy()

    newPoints = findColor(img, myColors)

    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)

    if len(myPoints)!=0:
        drawOnCanvas(myPoints,colorValues)

    if cv.waitKey(1) & 0xFF == ord('c'):
        myPoints.clear()

#   cv.imshow("Image", img)
    cv.imshow("Result", imgResult)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break