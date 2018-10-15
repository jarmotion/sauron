import cv2
import numpy as np
import time
#easily apply multiple image processes given a string

def initialize():
#storing results 
	global in_path
	global out_path
	global img_path
	img_path = "../data/check_images/2209967670.jpg"
	working_script = "img_order"
	working_time = time.strftime('%y%m%d_%H%M%S')
	working_folder = "../results/"
	in_path = working_folder + working_time + "_" + working_script + "-in.jpg"
	out_path = working_folder + working_time + "_" + working_script + "-out.jpg"

def procList(img, numstring = "1"):
	numlist = list(str(numstring))
	print(numlist)
	for effect in numlist:
		if effect == '1': 
			print("1 - denoising")
			img = denoise(img)
		elif effect == '2':
			print("2 - opening")
			img = opening(img)
		elif effect == '3': 
			img = erode(img)
			print("3 - erode")
		elif effect == '4':
			print("4 - threshold")
			img = threshold(img)
		elif effect == '5':
			print("5 - gaussianblur")
			img = gblur(img) 
	return(img)

#quickblur
def blur(in_img, blurtype = 2, sigmaX = 5, sigmaY = 5, bordertype = 0):
	#blur type: 1, 2:gaussian
	if blurtype == 2:
		out_img = cv2.GuassianBlur(in_img, (sigmaX, sigmaY), bordertype)
		return(out_img)

#quick defaults
def denoise(in_img, para1 = 60, para2 = 50, para3 = 5):
	return(cv2.fastNlMeansDenoising(in_img, None, para1, para2, para3))

def opening(in_img):
	kernel = np.ones((2,2), np.uint8)
	return(cv2.morphologyEx(in_img, cv2.MORPH_OPEN, kernel))

def erode(in_img):
	kernel = np.ones((2,2), np.uint8)
	return(cv2.erode(in_img, kernel, iterations = 1))

def threshold(in_img, para1 = 110, para2 = 255):
	retval, step = cv2.threshold(in_img, para1, para2, cv2.THRESH_BINARY)
	return(step)

def gblur(in_img):
	return(cv2.GaussianBlur(in_img,(5,5),0))

def peek(in_img):
	cv2.imshow('preview', in_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()