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
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

# lines = cv2.HoughLines(edges,5,2,50)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

minLineLength = 10
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

plt.imshow(img,cmap = 'gray')
plt.show()
# cv2.imwrite(out_path, img)
# cv2.imwrite(out_path,lines)