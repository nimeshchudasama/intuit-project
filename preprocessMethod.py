# preprocessMethod.py
# Runs a preprocessing method against the intermediate dataset and outputs them
# into the "preprocessed" folder
import numpy as np
import cv2
import time
import math
import os
from PIL import Image
import imutils
from skimage.filters import threshold_local
import matplotlib.pyplot as plt

def histogram_equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def runPreprocess(image_dir):
    imageCount = 100
    images = []
    tempImages = []
    times = []

    # Load intermediate set into images list
    for i in range(imageCount):
        images.append(str(image_dir) + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg")
    count = 0
    # Preprocess the images and store them in a temp list
    for i in range(len(images)):
        startTime = int(round(time.time() * 1000))
        # Open the image file
        img = cv2.imread(images[i])



        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((7, 7), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        area_thresh = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > area_thresh:
                area_thresh = area
                big_contour = c
        page = np.zeros_like(img)
        cv2.drawContours(page, [big_contour], 0, (255, 255, 255), -1)
        peri = cv2.arcLength(big_contour, True)
        corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

        result = img.copy()
        cv2.polylines(result, [corners], True, (0, 0, 255), 10, cv2.LINE_AA)

        # for simplicity get average of top/bottom side widths and average of left/right side heights
        # note: probably better to get average of horizontal lengths and of vertical lengths
        width = 0.5 * ((corners[0][0][0] - corners[1][0][0]) + (corners[3][0][0] - corners[2][0][0]))
        height = 0.5 * ((corners[2][0][1] - corners[1][0][1]) + (corners[3][0][1] - corners[0][0][1]))
        width = np.int0(width)
        height = np.int0(height)

        # reformat input corners to x,y list
        icorners = []
        for corner in corners:
            pt = [corner[0][0], corner[0][1]]
            icorners.append(pt)
        icorners = np.float32(icorners)

        # get corresponding output corners from width and height
        ocorners = [[width, 0], [0, 0], [0, height], [width, height]]
        ocorners = np.float32(ocorners)

        count += 1
        print("iteration" + str(count))
        print("Before" + str(len(ocorners)) + "||" + str(len(icorners)))

        while (len(ocorners) != 4 and len(icorners)!= 4):
            if len(ocorners) > 4:
                np.delete(ocorners,len(ocorners)-1)
            if len(icorners) > 4:
                np.delete(icorners,len(icorners)-1)
        # get perspective tranformation matrix
        print("After" + str(len(ocorners)) + "||" + str(len(icorners)))

        M = cv2.getPerspectiveTransform(icorners, ocorners)

        # do perspective
        warped = cv2.warpPerspective(img, M, (width, height))

        rgb_planes = cv2.split(warped)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_norm_planes)



        # The preprocesed images are saved temporarily in memory instead of written into output directory
        # so calculating the actual processing time won't be affected
        tempImages.append(result)

        # Record elapsed processing time for the image
        times.append(int(round(time.time() * 1000)) - startTime)

    print("Total processing time: ", sum(times), "ms")
    print("Average processing time: ", sum(times) / len(times), "ms")

    return tempImages


def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped, M

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
        cv2.imwrite(output_dir + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg",tempImage)

    print("Saved processed images to results directory")


main()
