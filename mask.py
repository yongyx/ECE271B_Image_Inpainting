import cv2
import torch
import random
import numpy as np

def createMask(img):
    channel, height, width = img.shape
    mask = np.full((height, width, channel), 255, np.uint8)
    for _ in range(np.random.randint(1,10)):
        #randomize start x locations
        x1, x2  =np.random.randint(1, height), np.random.randint(1,256)

        #randomize start y locations
        y1, y2 = np.random.randint(1, width), np.random.randint(1,256)

        #random thickness of line
        thickness = np.random.randint(15, 20)

        #draw black line
        cv2.line(mask, (x1, y1), (x2,y2), (0,0,0), thickness)


        #bitwise and with the mask
        print(img.transpose(1,2,0).shape)
        masked_img = cv2.bitwise_and(img.transpose(1,2,0), mask)
        masked_img[mask == 0] = 255
        masked_img = masked_img.transpose(2,0,1)
        mask = ~mask.transpose(2,0,1)

        return masked_img, mask
    

def rectangularMask(img):
    channel, height, width = img.shape
    mask = np.full((height, width, channel), 255, dtype=np.uint8)
    w, h = random.randint(1, width), random.randint(1, height)
    y, x = random.randint(0,height-h), random.randint(0, width-w)
    for i in range(channel):
        mask[y:y+h,x:x+w, i] = 0
    masked_img = cv2.bitwise_and(img.transpose(1,2,0), mask)
    masked_img[mask == 0] = 255
    masked_img = masked_img.transpose(2,0,1)
    mask = ~mask.transpose(2,0,1)
    return masked_img, mask