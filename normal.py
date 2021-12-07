from mediapipe.python import solutions
import numpy as np
import cv2


class remove_background_contour :
    def __init__ (self) ->None :
        self.blur        = 21
        self.canny_low   = 15
        self.canny_high  = 150
        self.min_area    = 0.0005
        self.max_area    = 0.95
        self.dilate_iter = 10
        self.erode_iter  = 10
        self.mask_color  = (0.0,0.0,0.0)

    def set_parameters(self, blur = 21, canny_low = 15, canny_high = 150, min_area = 0.0005, max_area = 0.95, dilate_iter = 10, erode_iter = 10, mask_color = (0.0,0.0,0.0)) :
        self.blur        = 21
        self.canny_low   = 15
        self.canny_high  = 150
        self.min_area    = 0.0005
        self.max_area    = 0.95
        self.dilate_iter = 10
        self.erode_iter  = 10
        self.mask_color  = (0.0,0.0,0.0)

    def solution(self, frame, bg_image ):
        height , width, channel = frame.shape
        bg_image = cv2.resize(bg_image, (width, height))
        # Convert image to grayscale        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, canny_low, canny_high)

        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        # get the contours and their areas
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

        # Get the area of the image as a comparison
        image_area = frame.shape[0] * frame.shape[1]  

        # calculate max and min areas in terms of pixels
        max_area = self.max_area * image_area
        min_area = self.min_area * image_area


        mask = np.zeros(edges.shape, dtype = np.uint8)

        for contour in contour_info:
           if contour[1] > min_area and contour[1] < max_area:
                mask = cv2.fillConvexPoly(mask, contour[0], (255))


        mask = cv2.dilate(mask, None, iterations=self.dilate_iter)
        mask = cv2.erode(mask, None, iterations=self.erode_iter)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        # return mask, mask
        mask_stack = np.dstack([mask]*3)
        mask_stack = mask_stack.astype('float32') / 255.0           
        frame = frame.astype('float32') / 255.0
        # output = np.where(mask_stack, frame, bg_image)
        masked = (mask_stack * frame) + ((1-mask_stack) * self.mask_color)
        masked = (masked * 255).astype('uint8')
        
        return mask_stack, masked
    pass

# Parameters

def nothing(x):
    pass
cv2.namedWindow('panel')

cv2.createTrackbar('Blur','panel',0,50, nothing)
cv2.createTrackbar('Min area','panel',0,1000, nothing)
cv2.createTrackbar('Max area','panel',0,1000, nothing)
cv2.createTrackbar('Canny low','panel',1,20, nothing)
cv2.createTrackbar('Canny high','panel',100,200, nothing)
cv2.createTrackbar('Dilate iter','panel',1,15, nothing)
cv2.createTrackbar('Erode iter','panel',1,15, nothing)

# initialize video from the webcam
video = cv2.VideoCapture(0)
bg_image =  cv2.imread('background.jpg')
rmbg = remove_background_contour()

while True:
    ret, frame = video.read()

    blur        = cv2.getTrackbarPos('Blur','panel')
    canny_low   = cv2.getTrackbarPos('Canny low','panel')
    canny_high  = cv2.getTrackbarPos('Canny high','panel')
    min_area    = cv2.getTrackbarPos('Min area','panel') / 1000.0
    max_area    = cv2.getTrackbarPos('Max area','panel') / 1000.0
    dilate_iter = cv2.getTrackbarPos('Dilate iter','panel')
    erode_iter  = cv2.getTrackbarPos('Erode iter','panel')
    
    rmbg.set_parameters(blur,canny_low,canny_high,min_area, max_area, dilate_iter, erode_iter)
    
    if ret == True:
        mask, output_image = rmbg.solution(frame, bg_image)
        cv2.imshow("Foreground", mask)
        cv2.imshow("Output", output_image)

        if cv2.waitKey(60) & 0xff == ord('q'):
            break
    else:
        break


cv2.destroyAllWindows()
video.release()