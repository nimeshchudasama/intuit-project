# preprocessMethod.py
# Runs a preprocessing method against the intermediate dataset and outputs them
# into the "preprocessed" folder

import time
import os
from PIL import Image


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
        tempImage = Image.open(images[i])
        
        # Do the preprocessing stuff here
        
        
        
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
        tempImage.save(output_dir + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg")
        
    print("Saved processed images to results directory")

main()
