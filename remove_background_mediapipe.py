# DataFlair background removal 

# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp

class remove_background_mediapipe:
    
    def __init__(self) -> None :
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.threshold = 0.6
    
    def set_parameters(self, threshold = 0.6) ->None :
        self.threshold = threshold
    
    def solution(self, frame, bg_image) :
        height , width, channel = frame.shape

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(RGB)

        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > self.threshold

        bg_image = cv2.resize(bg_image, (width, height))
        
        output = np.where(condition, frame, bg_image)
    
        return mask, output

def nothing(x):
    pass

image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

cv2.namedWindow('Output')
cv2.createTrackbar('Threshold'  ,'Output',0,100, nothing)
cv2.createTrackbar('Portrait'   ,'Output',0,50, nothing)

cap = cv2.VideoCapture(0)

rmbg = remove_background_mediapipe()

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    threshold = cv2.getTrackbarPos('Threshold','Output') / 100.0
    portrait  = cv2.getTrackbarPos('Portrait','Output') * 2 + 1

    rmbg.set_parameters(threshold)
    
    if  (portrait > 1):
        bg = cv2.GaussianBlur(frame, (portrait,portrait),0)
    else:
        bg = bg_image

    mask, output_image =  rmbg.solution(frame, bg)
    

    cv2.imshow("Output", output_image)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        image_index = (image_index + 1) % len(images)
        bg_image = cv2.imread(image_path+'/'+images[image_index])
    

cap.release()
cv2.destroyAllWindows()



