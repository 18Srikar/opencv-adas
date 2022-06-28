import cv2 as cv 
# OpenCV
import csv
# Comma seperated values file
import datetime
# For timestamping
from cv2 import destroyAllWindows
from tracker import *
tracker = DistTracker()
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# configuration file
frozen_model='frozen_inference_graph.pb'
# weights
model=cv.dnn_DetectionModel(frozen_model,config_file)
classNames=[]
names='names.txt'
# this file contains names of all detectable objects
with open(names,'r') as f:
    classNames=f.read().split("\n")
# print(classNames)
# print(len(classNames))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)
    # resizes a frame with given percentage, useful for downscaling high resoltion videos
vid=cv.VideoCapture("Video1.mp4")
if not vid.isOpened():
    vid=cv.VideoCapture(0)
    # if a valid path is not given webcam is chosen as the input
if not vid.isOpened():
    raise IOError("can't open")
font_scale=1.5
font=cv.FONT_HERSHEY_PLAIN
with open("data.csv", 'w') as f1:
    cwriter = csv.writer(f1)
    cwriter.writerow(['Timestamp','Object ID','Classification','Tracking Counter','Confidence'])
    while vid.isOpened():
        ret,frame=vid.read()
        frame50 = rescale_frame(frame, percent=50)
        detections=[]
        classIndex,confidence,bbox=model.detect(frame50,confThreshold=0.6)
        # Returns Class Indices, Confidence values and Bounding Boxes
        # Bounding boxes are x coordinate,y coordinate,width and height
        # Threshold is the minimum confidence value required for a detection. 
        # lower threshold results in more detections as well as false positives 
        if(len(classIndex)!=0):
            # condition to check whether detections have occured
            for classInd,conf,boxes in zip(classIndex,confidence,bbox):
                if(classInd<=80):
                    x,y,w,h=boxes[0],boxes[1],boxes[2],boxes[3]
                    boxes_ids = tracker.update([[[classNames[classInd-1]],[x,y,w,h],[conf]]])
                    for box_id in boxes_ids:
                        x,y,w,h,id,tc = box_id
                    cv.rectangle(frame50,boxes,(255,0,0),1)
                    # Draws a rectangle around the detections with the desired attributes
                    cv.putText(frame50,classNames[classInd-1],(boxes[0]+10,boxes[1]+40), font, font_scale,color=(0,255,0),thickness=2)
                    # Puts text in the rectangles with derired attributes
                    cwriter.writerow([str(datetime.datetime.now()),id+1,classNames[classInd-1],tc,conf])
                    # Writes data to the .csv file
                    print(bbox)
        cv.imshow('Video',frame50)
        if cv.waitKey(2)&0xFF==ord('c'):    
            break
        # Closes video when C key is pressed
f1.close()
vid.release()
cv.destroyAllWindows()
