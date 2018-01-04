#!/usr/bin/env python

import cv2








im = cv2.imread('2_1.jpg')

cv2.rectangle(im,(168,401),(272,470),(0,255,0),3)
cv2.rectangle(im,(5,440),(182,478),(0,255,0),3)
cv2.rectangle(im,(365,0),(512,447),(0,255,0),3)
cv2.rectangle(im,(0,0),(229,457),(0,255,0),3)
cv2.rectangle(im,(367,43),(489,437),(0,255,0),3)

cv2.imshow('src', im)
cv2.waitKey()

