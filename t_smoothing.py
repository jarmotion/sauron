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

img = cv2.imread(img_path)

blur = cv2.blur(img,(5,5))
gauss_blur = cv2.GaussianBlur(img,(5,5),0)
median_blur = cv2.medianBlur(img,5)
bilateral_blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gauss_blur),plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(median_blur),plt.title('Median Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(bilateral_blur),plt.title('Bilateral Blur')
plt.xticks([]), plt.yticks([])
plt.show()