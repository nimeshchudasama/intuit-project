# preprocessMethod.py
# Runs a preprocessing method against the intermediate dataset and outputs them
# into the "preprocessed" folder

# Denoising
# https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html

import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt


def runPreprocess(image_dir):
    imageCount = 100
    images = []
    tempImages = [] 
    times = []
    
    # Load intermediate set into images list
    for i in range(imageCount):
        images.append(str(image_dir) + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg")
        
        
    # Preprocess the images and store them in a temp list
    for i in range(len(images)):
        startTime = int(round(time.time() * 1000))
        # Open the image file
        tempImage = cv.imread(images[i])
        
        # Do the preprocessing stuff here
        tempImage = cv.fastNlMeansDenoisingColored(tempImage, None, 10, 10, 7, 21)
        
        # The preprocesed images are saved temporarily in memory instead of written into output directory
        # so calculating the actual processing time won't be affected
        tempImages.append(tempImage)
        
        # Record elapsed processing time for the image
        times.append(int(round(time.time() * 1000)) - startTime)
        
    print("Total processing time: ", sum(times), "ms")
    print("Average processing time: ", sum(times)/len(times), "ms")
        
    return tempImages
        

def main():
    image_dir = "intermediate"
    
    # Preprocess the images
    processedImages = runPreprocess(image_dir)
    
    
    # Output processed images into output directory
    output_dir = "results"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
        
    for i in range(len(processedImages)):
        tempImage = processedImages[i]
        cv.imwrite(output_dir + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg", tempImage)
        
    print("Saved processed images to results directory")

main()
