from lib import imgproc
import cv2

imgproc.initialize()
raw_img = cv2.imread(imgproc.img_path)

#1 - denoise
#2 - opening
#3 - erode
#4 - threshold
#5 - gaussian blur

numlist = 1523454

img = imgproc.procList(old_img, numlist)

imgproc.peek(raw_img)
imgproc.peek(img)