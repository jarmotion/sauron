import cv2
import numpy as np
from matplotlib import pyplot as plt

#storing results 
import time
img_path = "../data/check_images/2209967670.jpg"
working_script = "img_order"
working_time = time.strftime('%y%m%d_%H%M%S')
working_folder = "./results/"
in_path = working_folder + working_time + "_" + working_script + "-in.jpg"
out_path = working_folder + working_time + "_" + working_script + "-out.jpg"

kernel = np.ones((2,2),np.uint8)
col_img = cv2.imread(img_path)
img = cv2.imread(img_path,0)
step = img

#thresholding
# step = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#denoise
step = cv2.fastNlMeansDenoising(step, None, 60, 51, 5)

#erode
step = cv2.erode(step, kernel, iterations = 1)

#other thresholding
retval, step = cv2.threshold(step, 110, 255, cv2.THRESH_BINARY)

#opening
step = cv2.morphologyEx(step, cv2.MORPH_OPEN, kernel)

# #gaussianblur
step = cv2.GaussianBlur(step,(5,5),0)

#other thresholding
retval, step = cv2.threshold(step, 110, 255, cv2.THRESH_BINARY)

#find lines
# edges = cv2.Canny(step,50,150,apertureSize = 3)
# minLineLength = 400
# maxLineGap = 50
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for line_feature in lines:
# 	for x1,y1,x2,y2 in line_feature:
# 		cv2.line(col_img,(x1,y1),(x2,y2),(0,0,255),2)

plt.subplot(211),plt.imshow(col_img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(step),plt.title('Altered')
plt.xticks([]), plt.yticks([])
plt.show()

# cv2.imshow('image',step)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(in_path,col_img)
cv2.imwrite(out_path,step)