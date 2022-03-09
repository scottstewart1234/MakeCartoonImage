#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:49:07 2022

@author: Scott Stewart
License: MIT
"""

import cv2 
import numpy as np 

def make_cartoon_image(ImagePath, #input image path
                  cartoonPath, #output image path
                  AdaptiveThreshold1=25, #The lower this is the more lines appear
                  AdaptiveThreshold2=9, #The higher this is the less lines appear. must be less than AdaptiveThreshold1
                  ThickerLinesIterations = 3, #The higher this is the thicker lines will be
                  RemoveDotsIterations = 3,  #The higher this is the thinner lines will be. Removes some noisy dots as well
                  BlurIterations=5, #The higher this is the more cartoonish the image will look, but too high and it will all be one color
                  ValueIncrease = 1.5,#The higher this is the brigher the colors will be
                  shadow = 1.3, #The higher this is the more colorless the shadows will be
                  saturationIncrease = 2, #The higher this is the more cartoonish the colors will be
                  fade = 0.9, #The higher this is the darker shadows will be
                  buckets = 15): #The higher this is the more colors in the image.
    
    
    
    originalmage = cv2.imread(ImagePath)
    if originalmage is None:
        raise FileNotFoundError("Image Couldn't be found")
    
        
    #Create and blur greyscale image. This is used by the adaptive thresholding
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    #Create an adaptive threshold mask. This creates dark lines where there are objects
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, AdaptiveThreshold1, AdaptiveThreshold2)
    
    
    #Create the color image in order to blur it
    colorImage = originalmage
    for i in range(BlurIterations):
        #Blur the image. This will improve the results of the bucketing later
        colorImage = cv2.blur(colorImage, (15, 15))
    cartoonImage = colorImage
    #cartoonImage = cv2.bitwise_and(cartoonImage, cartoonImage, mask=getEdge)
    
    #Create a canny edge detection mask from the original image
    canny = cv2.Canny(originalmage, 150, 50,(5,5))
    canny = cv2.bitwise_not(canny)
    
    #Erode the inverse of the mask, this will make the lines more noticable 
    canny = cv2.erode(canny, (3,3), iterations=ThickerLinesIterations)
    
    #Combine the adaptive threshold and the canny masks
    canny = cv2.bitwise_and(canny, canny, mask=getEdge)
    
    #Dilate the results. This will get rid of a few extra dots.
    canny = cv2.dilate(canny, (3,3), iterations=RemoveDotsIterations)
    
    
    #Convert the image to hsv and remove the component parts
    hsv = cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    #Bucket the h values and assign each pixel in the bucket
    # the average value of each bucket. This creates
    # an effect where the the image has strong lines
    h_new = h.copy()
    amax = np.amax(h)
    for b in range(buckets):
        
       minimum = int(round(amax/buckets*b))
       maximum = int(round(amax/buckets*(b+1)))
       try:
           set_value = int(round(np.average(h, weights=((h >= minimum) & (h < maximum)))))
       except:
           #There is nothing in this bin.
           set_value = 0
       
       h_new = np.where((h >= minimum) & (h < maximum), set_value, h)
    
    #Modify the saturation of each pixel by a factor of saturationIncrease. if that 
    # is greater than the max value it becomes the max value. Otherwise it gets fadded by a
    #factor of fade
    #This brings out colors in the image making it look more cartoonish, except low saturation
    #areas which are even more colorless
    s = np.where(s*saturationIncrease >= 255, 255, s/fade)
    
    #Modify the value of each pixel by a factor of value increase. if that 
    # is greater than the max value it becomes the max value. Otherwise it gets fadded by a
    #factor of shadow
    #This brings out colors in the image making it look more cartoonish, except shadows
    #which are more dramatic. Makes more dramatic lines
    v = np.where(v*ValueIncrease >= 255, 255, v/shadow) #Switch these around for a cool effect
    
    #Recreate the hsv image
    hsv[:,:,0] = h_new
    hsv[:,:,1] = s
    hsv[:,:,2] = v
    
    #Convert back to bgr
    cartoonImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #Add the masks made earlier to the image
    cartoonImage = cv2.bitwise_and(cartoonImage, cartoonImage, mask=canny)
    #Save the image
    cv2.imwrite(cartoonPath, cartoonImage)
