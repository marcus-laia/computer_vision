import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if (self.results.multi_hand_landmarks):
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    #parametro para mudar cor = mpDraw.DrawingSpec(color=mpDraw.BLUE_COLOR)

        return img

    def findPosition(self, img, handNo=0, draw=[], markSize=5, boundingBoxDist=20):

        xList = []
        yList = []
        boundingBox = None
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                #print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                #print(id,draw,type(draw))
                if id in draw:
                    cv.circle(img, (cx, cy), markSize, (255, 0, 255), cv.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boundingBox = xmin-boundingBoxDist,ymin-boundingBoxDist,xmax+boundingBoxDist,ymax+boundingBoxDist

            if len(draw)==21:
                cv.rectangle(img,boundingBox[:2],boundingBox[2:],(0,255,0,2))

        return self.lmList, boundingBox

    def fingersUp(self):
        fingers = []
        # THUMB
        if self.lmList[4][1] > self.lmList[0][1]:
            if self.lmList[4][1] > self.lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lmList[4][1] < self.lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        # OTHER FINGERS
        for i in range(8, 21, 4):
            if self.lmList[i][2] < self.lmList[i - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self,p1,p2,img,draw=False):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if length < 50:
            centerColor = (0, 0, 255)
        elif length > 300:
            centerColor = (0, 255, 0)
        else:
            centerColor = (255,0,0)

        if draw:
            cv.circle(img, (x1, y1), 10, (255,0,0), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (255,0,0), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255,0,0), 3)
            cv.circle(img, (cx, cy), 10, centerColor, cv.FILLED)

        return length, img, [x1,y1,x2,y2,cx,cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()