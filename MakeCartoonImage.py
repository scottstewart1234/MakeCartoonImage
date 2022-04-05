#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:49:07 2022

@author: Scott Stewart
License: MIT
"""

import cv2 
import numpy as np 
import signal
import sys


def sigint_handler(signal, frame):
    print ('KeyboardInterrupt. Program Exiting')
    vral = False
    globals()["v_all"] = False
    #sys.exit(0)

def live_cartoon_image(originalImage, #input image path
                  BackgroundImage,
                  NewBackground,
                  AdaptiveThreshold1=25, #The lower this is the more lines appear
                  AdaptiveThreshold2=9, #The higher this is the less lines appear. must be less than AdaptiveThreshold1
                  ThickerLinesIterations = 5, #The higher this is the thicker lines will be
                  RemoveDotsIterations = 10,  #The higher this is the thinner lines will be. Removes some noisy dots as well
                  BlurIterations=2, #The higher this is the more cartoonish the image will look, but too high and it will all be one color
                  ValueIncrease = 1.5,#The higher this is the brigher the colors will be
                  shadow = 1.1, #The higher this is the more colorless the shadows will be
                  saturationIncrease = 2, #The higher this is the more cartoonish the colors will be
                  fade = 0.9, #The higher this is the darker shadows will be
                  buckets = 7): #The higher this is the more colors in the image. 
    
    #Convert the image to hsv and remove the component parts
    
    
    
    backSub = cv2.createBackgroundSubtractorKNN()
    for i in range(10):
        fgMask = backSub.apply(BackgroundImage)
    fgMask = backSub.apply(originalImage)
    
    #print(fgMask)
    fgMask = cv2.resize(fgMask, (originalImage.shape[1],originalImage.shape[0]))
    fgMask = np.where(fgMask > 250, 1, 0).astype(np.uint8)
    for i in range(BlurIterations):
    #Blur the image. This will improve the results of the bucketing later
        fgMask = cv2.blur(fgMask, (15, 15))
    
    fgMask = cv2.erode(fgMask, (5,5), iterations=ThickerLinesIterations)
    fgMask = cv2.dilate(fgMask, (3,3), iterations=RemoveDotsIterations)
    
    
    #maskedImage = cv2.bitwise_and(originalImage, originalImage, mask=fgMask)
    #background = cv2.bitwise_and(NewBackground, NewBackground, mask=cv2.bitwise_not(fgMask))
    #maskedImage += background

    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h_new = h.copy()
    s_new = s.copy()
    v_new = v.copy()
    grayScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    amax = np.amax(grayScaleImage)
    
    for b in range(buckets):
        
       minimum = int(round(amax/buckets*b))
       maximum = int(round(amax/buckets*(b+1)))
       try:
           set_value = int(round(np.average(h, weights=((grayScaleImage >= minimum) & (grayScaleImage < maximum)  & (fgMask == 1)))))
           
           set_values = int(round(np.average(s, weights=((grayScaleImage >= minimum) & (grayScaleImage < maximum) & (fgMask == 1)))))
           set_valuev = int(round(np.average(v, weights=((grayScaleImage >= minimum) & (grayScaleImage < maximum) & (fgMask == 1)))))
           
       except:
           #There is nothing in this bin.
           set_value = 0
           set_values = 0
           set_valuev = 0
       
       h_new = np.where((grayScaleImage >= minimum) & (grayScaleImage < maximum) & (fgMask == 1), h_new, h_new)
       s_new = np.where((grayScaleImage >= minimum) & (grayScaleImage < maximum) & (fgMask == 1), set_values, s_new)
       v_new = np.where((grayScaleImage >= minimum) & (grayScaleImage < maximum) & (fgMask == 1), set_valuev, v_new)
       
    
    hsv[:,:,0] = h_new
    hsv[:,:,1] = s_new
    hsv[:,:,2] = v_new
    #Convert back to bgr
    
    
    
    cartoonImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    canny = cv2.Canny(cartoonImage, 150, 50,(5,5))
    canny = cv2.bitwise_not(canny)
    cartoonImage = cv2.bitwise_and(cartoonImage, cartoonImage, mask=canny)
    
    
    maskedImage = cv2.bitwise_and(cartoonImage, cartoonImage, mask=fgMask)
    background = cv2.bitwise_and(NewBackground, NewBackground, mask=cv2.bitwise_not(fgMask))
    maskedImage += background
    maskedImage = cv2.bitwise_and(maskedImage, maskedImage, mask=canny)
    #Add the masks made earlier to the image
    #cartoonImage = cv2.bitwise_and(cartoonImage, cartoonImage, mask=canny)
    #Save the image
    #cv2.imwrite(cartoonPath, cartoonImage)
    return maskedImage
def getWebcam():
    
    new_background = cv2.imread("/home/scottstewart/Pictures/Green.png")
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(-1)
    FirstFrame = None
    fgMask = None
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        if(FirstFrame == None):
            FirstFrame = frame
            new_background = cv2.resize(new_background, (FirstFrame.shape[1],FirstFrame.shape[0]))
            
    else:
        rval = False
    
    while (rval and globals()["v_all"]):
       
        frame = np.flip(live_cartoon_image(frame,FirstFrame, new_background),1)
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(2)
        if(key == ord('`')):
            FirstFrame = frame
        if key == 27: # exit on ESC
            break
    
    vc.release()
    cv2.destroyWindow("preview")
 
globals()["v_all"] = True
signal.signal(signal.SIGINT, sigint_handler)
getWebcam()
    
#make_cartoon_image2("/home/scott/Downloads/Scotts.jpg","/home/scott/Downloads/Scott-CE3.jpg")