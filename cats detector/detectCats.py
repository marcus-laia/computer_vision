import cv2 as cv

frameWidth = 640
frameHeight = 480
catsCascade = cv.CascadeClassifier("./resources/haarcascade_frontalcatface_extended.xml")

img = cv.imread('./resources/cats.jpg')
img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2))

imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cats = catsCascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in cats:
    area = w*h
    if area>500:
        cv.rectangle(img, (x, y), (x + w, y + h), (255,0,255), 2)
        cv.putText(img,"Cat",(x,y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

cv.imshow("Result", img)
cv.waitKey(0)