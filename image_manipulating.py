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


kernel = np.ones((2,2),np.uint8)

img = cv2.imread(img_path)
step = img

step1 = img
plt.imshow(step1)
plt.show()

step = cv2.fastNlMeansDenoising(step, None, 50, 51, 5)

step2 = step
plt.imshow(step2)
plt.show()

step = cv2.erode(step, kernel, iterations = 2)

step3 = step
# plt.imshow(step)
# plt.show()

step = cv2.morphologyEx(step, cv2.MORPH_OPEN, kernel)

step4 = step
# plt.imshow(step)
# plt.show()

step = cv2.GaussianBlur(step,(5,5),0)
# step = cv2.Canny(step,50,150)

step5 = step
# plt.imshow(step)
# plt.show()

plt.subplot(231),plt.imshow(step1),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(step2),plt.title('Step 1')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(step3),plt.title('Step 2')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(step4),plt.title('Step 3')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(step5),plt.title('Step 4')
plt.xticks([]), plt.yticks([])
plt.show()