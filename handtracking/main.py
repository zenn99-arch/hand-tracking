from sre_constants import SUCCESS
import cv2
import mediapipe
import time

cap = cv2.VideoCapture(0)

mphands = mp.solution.hands
hands = mphands.hands() 

while True:
    SUCCESS, img = cap.read()


    cv2.imshow("image", img)
    cv2.waitKey(1)