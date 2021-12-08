import cv2
import numpy as np
import sys
import os
class remove_background_subtract :
	def __init__(self) -> None :
		self.threshold_chanel = 13
		self.threshold_gray   = 10
		self.blur			  = 3
		
	def resize(self,dst,img):
		width = img.shape[1]
		height = img.shape[0]
		dim = (width, height)
		resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
		return resized

	def set_parameters(self, threshold_chanel = 13, threshold_gray = 10, blur = 3) ->None:
		self.threshold_chanel = threshold_chanel
		self.threshold_gray   = threshold_gray
		self.blur 			  = blur

	def solution(self,img, ref_img, bg) :
		bg = self.resize(bg,ref_img)
		
		diff = cv2.absdiff(ref_img,img)
	
		diff[abs(diff)<self.threshold_chanel]=0

		gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		gray[np.abs(gray) < self.threshold_gray] = 0
		gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
		
		fgmask = gray.astype(np.uint8)
		fgmask[fgmask>0]=255
		fgmask_inv = cv2.bitwise_not(fgmask)
	
		fgimg = cv2.bitwise_and(img,img,mask = fgmask)
		bgimg = cv2.bitwise_and(bg,bg,mask = fgmask_inv)
		
		output_image = cv2.add(bgimg,fgimg)
		return fgmask,output_image

def nothing(x):
    pass

cv2.namedWindow('Output')

cv2.createTrackbar('Threshold chanel','Output',1,255, nothing)
cv2.createTrackbar('Threshold gray'  ,'Output',1,255, nothing)
cv2.createTrackbar('Blur'  ,'Output',0,20, nothing)
cv2.createTrackbar('Portrait'   ,'Output',0,50, nothing)

video = cv2.VideoCapture(0)
success, ref_frame = video.read()
# video_background = cv2.VideoCapture("ocean.mp4")

image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

rmbg = remove_background_subtract()

while(True):
	success, frame = video.read()
	frame = cv2.flip(frame, 1)

	threshold_chanel = cv2.getTrackbarPos('Threshold chanel','Output')
	threshold_gray   = cv2.getTrackbarPos('Threshold gray','Output')
	blur             = cv2.getTrackbarPos('Blur','Output') * 2 + 1
	portrait 		 = cv2.getTrackbarPos('Portrait','Output') * 2 + 1

	# success2, bg = video_background.read()
	# if not success2 : 
	# 	video_background.release()
	# 	video_background = cv2.VideoCapture("ocean.mp4")
	# 	success2, bg = video_background.read()
	
	if  (portrait > 1):
		bg = cv2.GaussianBlur(frame, (portrait,portrait),0)
	else:
		bg = bg_image
	
	rmbg.set_parameters(threshold_chanel, threshold_gray, blur)

	mask, output_image = rmbg.solution(frame, ref_frame, bg)
	
	cv2.imshow("Mask", mask)
	cv2.imshow("Output", output_image)
		
	key = cv2.waitKey(5) & 0xFF

	if ord('q') == key:
		break

	elif ord('p') == key:
		ref_frame = frame
	elif key == ord('d'):
		image_index = (image_index + 1) % len(images)
		bg_image = cv2.imread(image_path+'/'+images[image_index])
	

	

cv2.destroyAllWindows()
video.release()
video_background.release()
