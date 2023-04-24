import cv2 
import matplotlib.pyplot as plt

path = './data/traffic.avi'
# read the input video or input images sequence
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(path))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


## background initialization
# generate the foreground mask using GMM
backSub = cv2.createBackgroundSubtractorMOG2()


while True:
    cv2.waitKey(100)
    ret, frame = capture.read()
    if frame is None:
        break
    
    ## update the background model
    fgMask = backSub.apply(frame)
    
    ## get the frame number and write it on the current frame
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    ## show the current frame and the fg masks
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cv2.destroyAllWindows()
