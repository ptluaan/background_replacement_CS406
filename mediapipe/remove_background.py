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
    
    def solution(self, frame, bg_image) :
        
        height , width, channel = frame.shape

        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.selfie_segmentation.process(RGB)

        mask = results.segmentation_mask
        
        condition = np.stack((mask,) * 3, axis=-1) > 0.6

        bg_image = cv2.resize(bg_image, (width, height))
        
        output = np.where(condition, frame, bg_image)
    
        return mask, output


image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

cap = cv2.VideoCapture(0)

rmbg = remove_background_mediapipe()

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    mask, output_image =  rmbg.solution(frame, bg_image)

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



