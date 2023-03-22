#!/usr/bin/python

import sys, os
import cv2
import numpy as np
import argparse
from Inpainter import Inpainter
from mask import createMask

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    """
    Usage: python main.py pathOfInputImage pathOfMaskImage [,halfPatchWidth=4].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgRoot', default='./BG_20k/train', help='name of image to perform inpainting')
    parser.add_argument('--maskRoot', default='./', help='root of mask image')
    parser.add_argument('--resultName', default='result.jpg', help='name of inpainted image for saving')

    opt = parser.parse_args()
    print(opt)
    # if not len(sys.argv) == 3 and not len(sys.argv) == 4:
    #     print ('Usage: python main.py pathOfInputImage pathOfMaskImage [,halfPatchWidth].')
    #     exit(-1)
    #
    # if len(sys.argv) == 3:
    #     halfPatchWidth = 4
    # elif len(sys.argv) == 4:
    #     try:
    #         halfPatchWidth = int(sys.argv[3])
    #     except ValueError:
    #         print('Unexpected error:', sys.exc_info()[0])
    #         exit(-1)
    #
    halfPatchWidth = 4
    # image File Name
    print(opt.imgRoot)
    originalImage = cv2.imread(opt.imgRoot, 1)
    # originalImage = cv2.resize(originalImage,(1000,1000))
    # cv2.imshow('w', originalImage)
    # CV_LOAD_IMAGE_COLOR: loads the image in the RGB format TODO: check RGB sequence
    # originalImage = cv2.imread(imageName, cv2.CV_LOAD_IMAGE_COLOR)
    if originalImage is None:
        print('Error: Unable to open Input image.')
        exit(-1)
    
    # mask File Name
    inpaintMask = cv2.imread(opt.maskRoot, 0)
    # inpaintMask = cv2.resize(inpaintMask, (1000, 1000))
    # inpaintMask = cv2.imread(maskName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if inpaintMask is None:
        print('Error: Unable to open Mask image.')
        exit(-1)
    
    i = Inpainter(originalImage, inpaintMask, halfPatchWidth)
    if i.checkValidInputs()== i.CHECK_VALID:
        i.inpaint()
        cv2.imwrite(opt.resultName, i.result)
        # cv2.namedWindow("result")
        # cv2.imshow("result", i.result)
        # cv2.waitKey()
    else:
        print('Error: invalid parameters.')
    
