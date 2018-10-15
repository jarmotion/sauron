import cv2
import numpy as np
from matplotlib import pyplot as plt

#storing results 
import time
img_path = "../data/check_images/2189834730.jpg"
working_script = "houghlines"
working_time = time.strftime('%y%m%d_%H%M%S')
working_folder = "./results/"
in_path = working_folder + working_time + "_" + working_script + "-in.jpg"
out_path = working_folder + working_time + "_" + working_script + "-out.jpg"

img = cv2.imread(img_path, 0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()