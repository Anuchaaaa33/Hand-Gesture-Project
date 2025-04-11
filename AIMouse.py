import cv2 as cv
import numpy as np
import math
import time
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=3, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon*100)
        self.trackCon =  int(trackCon*100)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.tipIDS = [4, 8, 12, 16, 20]

def findHands(self, img, draw=True):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    self.results = self.hands.process(img_rgb)
    if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    return img
    
def findPosition(self, img, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []
    if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  #converting them into pixels
                xList.append(cx)   
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])  ##adding them to the list
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)  ##5 is radius, the other parameter id color
                ##bounding box 
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
 
            if draw:
                cv.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
                return self.lmList, bbox

def fingerUP(self):
    fingers =[]
    #thumb
    if self.lmList[self.tipIDS[0]][1] > self.lmList[self.tipIDS[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
 
    # Fingers
    for id in range(1, 5):
 
        if self.lmList[self.tipIDS[id]][2] < self.lmList[self.tipIDS[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
 
        # totalFingers = fingers.count(1)
 
    return fingers

def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    
    detecter = handDetector()
    cap = cv.VideoCapture(0)
    cam_w, cam_h = 648, 480
    cap.set(3, cam_w)
    cap.set(4, cam_h)
    while True:
        __, img = cap.read()
        img = cv.flip(img, 1)
        img = detecter.findHands(img)
        lmList, bbox = detecter.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv.imshow("Image", img)

        if cv.waitKey(1) & 0xff == ord('q'):
            break

cv.destroyAllWindows()