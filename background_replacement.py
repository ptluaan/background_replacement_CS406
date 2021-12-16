import cv2
import numpy as np
import mediapipe as mp

class background_replacement_mediapipe:
    
    def __init__(self) -> None :
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.threshold = 0.6
    
    def set_parameters(self, threshold = 0.6) ->None :
        self.threshold = threshold
    
    def solution(self, frame, bg_image) :
        height , width, channel = frame.shape
        bg_image = cv2.resize(bg_image, (width, height))

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.selfie_segmentation.process(RGB)
        mask = results.segmentation_mask

        mask_img = np.zeros((height,width), np.uint8)
        mask_img[mask > self.threshold] = 255

        mask_img_inv = cv2.bitwise_not(mask_img)
        
        foreground = cv2.bitwise_and(frame, frame, mask= mask_img)
        background = cv2.bitwise_and(bg_image, bg_image, mask= mask_img_inv)
        
        output_image = cv2.add(foreground, background)

        return mask_img, output_image

class background_replacement_contours :
    def __init__ (self) ->None :
        self.blur        = 21
        self.canny_low   = 15
        self.canny_high  = 150
        self.min_area    = 0.05
        self.max_area    = 0.95
        self.dilate_iter = 10
        self.erode_iter  = 10

    def set_parameters(self, blur = 21, canny_low = 15, canny_high = 150, min_area = 0.0005, max_area = 0.95, dilate_iter = 10, erode_iter = 10) :
        self.blur        = blur
        self.canny_low   = canny_low
        self.canny_high  = canny_high
        self.min_area    = min_area
        self.max_area    = max_area
        self.dilate_iter = dilate_iter
        self.erode_iter  = erode_iter
      

    def solution(self, frame, bg_image ):
        height , width, channel = frame.shape
        bg_image = cv2.resize(bg_image, (width, height))
       
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
        edges = cv2.Canny(image_gray, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contours_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
   
        image_area = frame.shape[0] * frame.shape[1]  

        max_area = self.max_area * image_area
        min_area = self.min_area * image_area

        mask = np.zeros(edges.shape, dtype = np.uint8)

        for contour in contours_info:
           if contour[1] > min_area and contour[1] < max_area:
                mask = cv2.fillConvexPoly(mask, contour[0], (255))

        mask = cv2.dilate(mask, None, iterations=self.dilate_iter)
        mask = cv2.erode(mask, None, iterations=self.erode_iter)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        
        mask[mask>0] = 255
        mask_inv = cv2.bitwise_not(mask)

        foreground = cv2.bitwise_and(frame, frame, mask= mask)
        background = cv2.bitwise_and(bg_image, bg_image, mask= mask_inv)

        output_image = cv2.add(foreground, background)
        
        return mask, output_image

class background_replacement_subtract :
	def __init__(self) -> None :
		self.threshold_chanel = 13
		self.threshold_gray   = 10
		self.blur			  = 1
		
	def resize(self,dst,img):
		width = img.shape[1]
		height = img.shape[0]
		dim = (width, height)
		resized = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)
		return resized

	def set_parameters(self, threshold_chanel = 13, threshold_gray = 10, blur = 1) ->None:
		self.threshold_chanel = threshold_chanel
		self.threshold_gray   = threshold_gray
		self.blur 			  = blur

	def solution(self, img, ref_img, bg) :
		bg = self.resize(bg,img)
		
		diff = cv2.absdiff(ref_img,img)
		diff[abs(diff)<self.threshold_chanel]=0

		gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
		gray[np.abs(gray) < self.threshold_gray] = 0

		mask = gray.astype(np.uint8)
		mask[mask>0]=255
		mask_inv = cv2.bitwise_not(mask)
	
		foreground = cv2.bitwise_and(img,img,mask = mask)
		background = cv2.bitwise_and(bg,bg,mask = mask_inv)
		
		output_image = cv2.add(foreground, background)
		return mask, output_image

