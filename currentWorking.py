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


kernel = np.ones((5,5),np.uint8)



org_img = cv2.imread(img_path)
img = org_img
#applying gaussian blur to have more features to identify and remove noise
step_1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
step_2 = cv2.GaussianBlur(step_1,(5,5),0)
step_3 = cv2.Canny(step_2,50,150)


last_step = step_3
edges = last_step


if(True):
	lines = cv2.HoughLines(edges,1,np.pi/180,600)
	for line_feature in lines:
		for rho,theta in line_feature:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1500*(-b))
		    y2 = int(y0 - 1000*(a))
		    cv2.line(img,(x1,y1-1),(x2,y2-1),(255,255,255),2)
		    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
		    cv2.line(img,(x1,y1+1),(x2,y2+1),(255,255,255),2)

if(True):
	minLineLength = 200
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
	for line_feature in lines:
		for x1,y1,x2,y2 in line_feature:
			cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)



plt.subplot(221),plt.imshow(org_img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(step_1),plt.title('Step 1')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(step_2),plt.title('Step 2')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img),plt.title('Lines')
plt.xticks([]), plt.yticks([])

plt.show()
