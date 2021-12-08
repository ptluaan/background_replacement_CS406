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
        # cv2.imshow("e",edges)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        # cv2.imshow("nw",edges)
        
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

       
        image_area = frame.shape[0] * frame.shape[1]  


        max_area = self.max_area * image_area
        min_area = self.min_area * image_area

        mask = np.zeros(edges.shape, dtype = np.uint8)

        for contour in contour_info:
           if contour[1] > min_area and contour[1] < max_area:
                mask = cv2.fillConvexPoly(mask, contour[0], (255))

        mask = cv2.dilate(mask, None, iterations=self.dilate_iter)
        mask = cv2.erode(mask, None, iterations=self.erode_iter)
        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        
        mask[mask>0] = 255

        mask_inv = cv2.bitwise_not(mask)

        fg_imgae = cv2.bitwise_and(frame, frame, mask= mask)
        bg_image = cv2.bitwise_and(bg_image, bg_image, mask= mask_inv)

        output_image = cv2.add(fg_imgae, bg_image)
        
       
        # mask_stack = np.dstack([mask]*3)
        # mask_stack = mask_stack.astype('float32') / 255.0           
        # frame = frame.astype('float32') / 255.0
    
        # masked = (mask_stack * frame) + ((1-mask_stack) * self.mask_color)
        # masked = (masked * 255).astype('uint8')
       
        # cv2.imshow("masked", masked)
        return mask, output_image
        
    pass

# Parameters

def nothing(x):
    pass
    
cv2.namedWindow('Output')

cv2.createTrackbar('Blur','Output',0,25, nothing)
cv2.createTrackbar('Min area','Output',0,200, nothing)
cv2.createTrackbar('Max area','Output',0,1000, nothing)
cv2.createTrackbar('Canny low','Output',1,20, nothing)
cv2.createTrackbar('Canny high','Output',100,200, nothing)
cv2.createTrackbar('Dilate iter','Output',1,15, nothing)
cv2.createTrackbar('Erode iter','Output',1,15, nothing)

# initialize video from the webcam
video = cv2.VideoCapture(0)
bg_image =  cv2.imread('background.jpg')
rmbg = remove_background_contour()

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    blur        = cv2.getTrackbarPos('Blur','Output') * 2 + 1
    canny_low   = cv2.getTrackbarPos('Canny low','Output')
    canny_high  = cv2.getTrackbarPos('Canny high','Output')
    min_area    = cv2.getTrackbarPos('Min area','Output') / 1000.0
    max_area    = cv2.getTrackbarPos('Max area','Output') / 1000.0
    dilate_iter = cv2.getTrackbarPos('Dilate iter','Output')
    erode_iter  = cv2.getTrackbarPos('Erode iter','Output')
    
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