import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#storing results 
img_path = "../data/check_images/2189834730.jpg"
working_script = "canny"
working_time = time.strftime('%y%m%d_%H%M%S')
working_folder = "./results/"
in_path = working_folder + working_time + "_" + working_script + "-in.jpg"
out_path = working_folder + working_time + "_" + working_script + "-out.jpg"

img = cv2.imread(img_path,0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

cv2.imwrite(in_path, img)
cv2.imwrite(out_path,edges)


plt.show()