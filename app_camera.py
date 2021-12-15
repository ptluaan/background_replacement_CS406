from mediapipe.python import solutions
from background_replacement import *
import os

solution = 0

flag_mediapipe = False
flag_contours  = False
flag_subtract  = False

flag_flip = False

flag_mask = False

image_path = 'images'
images = os.listdir(image_path)

image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

def nothing(x):
    pass

# create solutions
br_mediapipe = background_replacement_mediapipe()
br_contours  = background_replacement_contours()
br_subtract  = background_replacement_subtract()

# camera
camera = cv2.VideoCapture(0)
_,ref_frame = camera.read()

# setting panel
cv2.namedWindow('Panel')

# Pannel for background_replacement_mediapipe
def panel_mediapipe():
    cv2.destroyWindow('Panel')
    cv2.namedWindow('Panel')

    cv2.createTrackbar('Threshold'  ,'Panel',0,200, nothing)
    cv2.createTrackbar('Portrait'   ,'Panel',0,50, nothing)

# Pannel for background_replacement_contours
def panel_contours():
    cv2.destroyWindow('Panel')
    cv2.namedWindow('Panel',cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Portrait'   ,'Panel',0,50, nothing)
    cv2.createTrackbar('Blur'       ,'Panel',0,25, nothing)
    cv2.createTrackbar('Min area'   ,'Panel',0,700, nothing)
    cv2.createTrackbar('Max area'   ,'Panel',0,1000, nothing)
    cv2.createTrackbar('Canny low'  ,'Panel',1,250, nothing)
    cv2.createTrackbar('Canny high' ,'Panel',1,250, nothing)
    cv2.createTrackbar('Dilate iter','Panel',1,15, nothing)
    cv2.createTrackbar('Erode iter' ,'Panel',1,15, nothing)

# Pannel for background_replacement_subtract
def panel_subtract():
    cv2.destroyWindow('Panel')
    cv2.namedWindow('Panel')

    cv2.createTrackbar('Threshold chanel','Panel',1,255, nothing)
    cv2.createTrackbar('Threshold gray'  ,'Panel',1,255, nothing)
    cv2.createTrackbar('Blur'            ,'Panel',0,20, nothing)
    cv2.createTrackbar('Portrait'        ,'Panel',0,50, nothing)

def mediapipe_handle(frame, background) :
    global flag_mediapipe, flag_contours, flag_subtract, flag_flip, camera

    if not flag_mediapipe : panel_mediapipe()

    flag_mediapipe = True
    flag_contours  = False
    flag_subtract  = False

    threshold = cv2.getTrackbarPos('Threshold','Panel') / 100.0
    portrait  = cv2.getTrackbarPos('Portrait' ,'Panel') * 2 + 1

    br_mediapipe.set_parameters(threshold)
    if  (portrait > 1):
        bg = cv2.GaussianBlur(frame, (portrait,portrait),0)
    else:
        bg = background
    return br_mediapipe.solution(frame, bg)

def contours_handle(frame, background) :
    global flag_mediapipe, flag_contours, flag_subtract, flag_flip, camera

    if not flag_contours : panel_contours()

    flag_contours  = True
    flag_mediapipe = False
    flag_subtract  = False

    blur        = cv2.getTrackbarPos('Blur'       ,'Panel') * 2 + 1
    canny_low   = cv2.getTrackbarPos('Canny low'  ,'Panel')
    canny_high  = cv2.getTrackbarPos('Canny high' ,'Panel')
    min_area    = cv2.getTrackbarPos('Min area'   ,'Panel') / 1000.0
    max_area    = cv2.getTrackbarPos('Max area'   ,'Panel') / 1000.0
    dilate_iter = cv2.getTrackbarPos('Dilate iter','Panel')
    erode_iter  = cv2.getTrackbarPos('Erode iter' ,'Panel')
    portrait    = cv2.getTrackbarPos('Portrait'   ,'Panel') * 2 + 1

    br_contours.set_parameters(blur,canny_low,canny_high,min_area, max_area, dilate_iter, erode_iter)
    
    if  (portrait > 1):
        bg = cv2.GaussianBlur(frame, (portrait,portrait),0)
    else:
        bg = background
    return br_contours.solution(frame, bg)

def subtract_handle(frame, ref_frame, background) :
    global flag_mediapipe, flag_contours, flag_subtract, flag_flip, camera

    if not flag_subtract : panel_subtract()

    flag_subtract  = True
    flag_mediapipe = False
    flag_contours  = False

    threshold_chanel = cv2.getTrackbarPos('Threshold chanel','Panel')
    threshold_gray   = cv2.getTrackbarPos('Threshold gray'  ,'Panel')
    blur             = cv2.getTrackbarPos('Blur'            ,'Panel') * 2 + 1
    portrait 		 = cv2.getTrackbarPos('Portrait'        ,'Panel') * 2 + 1

    br_subtract.set_parameters(threshold_chanel, threshold_gray, blur)

    if  (portrait > 1):
        bg = cv2.GaussianBlur(frame, (portrait,portrait),0)
    else:
        bg = background

    return br_subtract.solution(frame, ref_frame, bg)

while True:
    # try :
    key = cv2.waitKey(1) & 0xFF
    
    _, frame = camera.read()

    if  key == ord('f') :
        flag_flip ^= True
    
    if (flag_flip) :
        frame = cv2.flip(frame,1)

    if   key == ord('q') or key == 27:
        break
    
    elif key == ord('d'):
        image_index = (image_index + 1) % len(images)
        bg_image = cv2.imread(image_path+'/'+images[image_index])
    elif ord('p') == key:
        ref_frame = frame
    


    elif key == ord('1'):
        solution = 0
    elif key == ord('2'):
        solution = 1
    elif key == ord('3'):
        solution = 2

    if   0 == solution :
        mask, output = mediapipe_handle(frame, bg_image)
    elif 1 == solution :
        mask, output = contours_handle(frame, bg_image)
    else :
        mask, output = subtract_handle(frame, ref_frame, bg_image)
    
    if key == ord('m'):
        if flag_mask :
            cv2.destroyWindow('mask')
        flag_mask ^= True

    if flag_mask :
        cv2.imshow('mask',mask)
   
    cv2.imshow("Output", output)

    if ord('s') == key:
        out_path = 'Output'
        out = os.listdir(out_path)
        name_out = str(len(out)) 
        cv2.imwrite(out_path +"/" + name_out + ".jpg",output)
    # except :
    #     pass

cv2.destroyAllWindows()
camera.release()