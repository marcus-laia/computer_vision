import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

while True:
    success, img = cap.read()

    for barcode in decode(img):
        data = barcode.data.decode('utf-8')
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,(255,0,255),3)
        ret = barcode.rect
        cv.putText(img,data,(ret[0],ret[1]),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)

    cv.imshow('Result', img)
    cv.waitKey(1)

