import cv2 as cv
import numpy as np

#widthImg = 640
#heightImg = 480
# widthImg = 960
# heightImg = 720
# paperWidth = 432
# paperHeight = 611

widthImg = 1500
heightImg = 2000
paperWidth = 552
paperHeight = 780

images = [cv.imread('resources/doc1.jpg'),
          cv.imread('resources/doc2.jpg')]


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def preProcessing(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDilated = cv.dilate(imgCanny,kernel,iterations=2)
    imgEroded = cv.erode(imgDilated,kernel,iterations=1)

    return imgEroded


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>5000:
            #cv.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv.drawContours(imgContour, biggest, -1, (255, 0, 0), 50)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[paperWidth,0],[0,paperHeight],[paperWidth,paperHeight]])
    matrix= cv.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv.warpPerspective(img, matrix, (paperWidth,paperHeight))

    borderToRemove = 10
    imgCropped = imgOutput[borderToRemove:imgOutput.shape[0]-borderToRemove, borderToRemove:imgOutput.shape[1]-borderToRemove]
    imgCropped = cv.resize(imgCropped,(paperWidth,paperHeight))

    return imgCropped
index = 0
for img in images:

    img = cv.resize(img,(widthImg,heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)

    if biggest.size != 0:
        imgWarped = getWarp(img,biggest)
        imageArray = ([img,imgThres],
                    [imgContour,imgWarped])
        cv.imshow(("Document "+str(index)), imgWarped)
    else:
        imageArray = ([img, imgThres],
                      [img, img])

    stackedImages = stackImages(0.19,imageArray)

    cv.imshow(("Process "+str(index)), stackedImages)
    #cv.imwrite("result.jpg", imgWarped)

    index+=1

cv.waitKey(0)