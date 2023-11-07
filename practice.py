import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import dlib
import time

roadLimit=70
 
cap=cv2.VideoCapture("../Videos/cars.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
model=YOLO("../yolov8l.pt")

def estimateSpeed(location1, location2,id,fps=int(cap.get(cv2.CAP_PROP_FPS))):
    d_pixels = math.sqrt(math.pow(location2[id][1] - location1[id][1], 2) + math.pow(location2[id][0] - location1[id][0], 2))
    ppm = 8.8
    d_meters = (d_pixels)*240 / (ppm*location1[id][1]*3)
    speed = d_meters * fps * 3.6
    return speed

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (740,540),interpolation=cv2.INTER_AREA)
# cv2.imshow("Image",mask)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [185, 250, 380, 250]
totalCount = []
carLocation1 = {}
carLocation2 = {}

while True:
    suc,frame=cap.read();
    if not suc:
        print("Error: Could not read frame.")
        break
    frame=cv2.resize(frame,(740,540),interpolation=cv2.INTER_AREA)
    imgGraphics = cv2.imread("graphics1.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, imgGraphics, (0, 0))
    imgRegion=cv2.bitwise_and(mask,frame)
    results=model(imgRegion,stream=True)
    detections = np.empty((0, 5))