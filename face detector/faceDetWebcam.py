import cv2 as cv

faceCascade = cv.CascadeClassifier("../haarcascades/haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

while True:
    success, wc = cap.read()
    wcGray = cv.cvtColor(wc,cv.COLOR_BGR2GRAY)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    faces = faceCascade.detectMultiScale(wcGray, 1.1, 4)

    for (x, y, w, h) in faces:
        
        cv.rectangle(wc, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("Video", wc)

cv.waitKey(0)