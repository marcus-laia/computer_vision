import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

with open('resources/data_file.txt') as f:
    dataList = f.read().splitlines()

while True:
    success, img = cap.read()

    for barcode in decode(img):
        data = barcode.data.decode('utf-8')

        if data in dataList:
            output = 'Authorized'
            color = (0,255,0)
        else:
            output = 'Un-authorized'
            color = (0,0,255)

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,color,3)
        ret = barcode.rect
        cv.putText(img,output,(ret[0],ret[1]),cv.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv.imshow('Result', img)
    cv.waitKey(1)
