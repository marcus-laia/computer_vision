import cv2 as cv

frameWidth = 640
frameHeight = 480
catsCascade = cv.CascadeClassifier("resources/haarcascade_frontalcatface_extended.xml")

cap = cv.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,100)

savedImgs = 0

while True:
    success, video = cap.read()
    img = video.copy()

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cats = catsCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in cats:
        area = w*h
        if area>500:
            cv.rectangle(img, (x, y), (x + w, y + h), (255,0,255), 2)
            cv.putText(img,"Cat",(x,y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

            #cv.imshow("ROI", imgRoi)

    if cv.waitKey(1) & 0xFF == ord('s'):
        for (x, y, w, h) in cats:
            area = w * h
            if area > 500:
                savedImgs += 1
                imgRoi = video[y:y+h, x:x+w]
                cv.imwrite(('savedImages/cat_'+str(savedImgs)+'.jpg'),imgRoi)
                cv.imshow(str(savedImgs),(cv.imread('savedImages/cat_'+str(savedImgs)+'.jpg')))

    cv.imshow("Result", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break